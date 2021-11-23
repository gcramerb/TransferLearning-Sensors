import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier,classifierTest
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss, classDistance
#import geomloss

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict


"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""
class TLmodel(LightningModule):
	
	def __init__(
			self,
			lr_source: float = 0.002,
			lr_target: float = 0.002,
			batch_size: int = 128,
			n_classes: int = 6,
			alphaS: float = 1.0,
			betaS: float = 0.5,
			alphaT: float = 1.0,
			penalty: str = 'mmd',
			data_shape: tuple = (1,50,6),
			modelHyp: dict = None,
			FeName: str = 'fe1',
			max_eph_S: int = 40,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()

        # networks
		self.clf = classifier(n_classes = self.hparams.n_classes,
		                      FeName=self.hparams.FeName,
		                      hyp=self.hparams.modelHyp,
		                      inputShape=self.hparams.data_shape)
		
		self.AE = ConvAutoencoder(FeName = self.hparams.FeName,
		                          hyp = self.hparams.modelHyp,
		                          inputShape = self.hparams.data_shape)
		
		self.clf.build()
		self.AE.build()
		self.test_metrics = []
		self.train_clf = True
		# for ae in self.AE.parameters():
		# 	ae.requires_grad = False
		
		# from torchsummary import summary
		# summary(self.AE.to('cuda'), (1,50,6))
		
		#SET THE losses:
		self.recLoss = torch.nn.MSELoss()
		self.clfLoss = torch.nn.CrossEntropyLoss()
		if self.hparams.penalty == 'mmd':
			self.discLoss = MMDLoss()
		elif self.hparams.penalty == 'ot':
			self.discLoss = OTLoss()
		else:
			raise ValueError('specify a valid discrepancy loss!')
		
		self.clDist = classDistance()
		self.modelName = ['Classifier', 'Reconstructior']
		self.datasetName = ['Source','Target']
		

	def forward(self, X):
		return self.clf(X)
	def _get_metrics(self,labSource,predSource,labTarget,predTarget,AEloss = None,clfLoss = None):
		
		
		accSource = accuracy_score(labSource.cpu().numpy(), np.argmax(predSource.cpu().numpy(), axis=1))
		accTarget = accuracy_score(labTarget.cpu().numpy(), np.argmax(predTarget.cpu().numpy(), axis=1))
		if AEloss is not None:
			metrics = {'AEloss': AEloss.item(),
			           'clfLoss': clfLoss.item(),
			           'accSource': accSource,
			           'accTarget': accTarget
			           }
		else:
			metrics = {'accSource': accSource,
			           'accTarget': accTarget
			           }
		return metrics
		
		
	def _shared_eval_step(self, batch,stage = 'val'):

		source,target = batch[0],batch[1]
		dataSource,labSource = source['data'],source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()

		latentS, predS = self.clf(dataSource)
		latentT,rec = self.AE.forward(dataTarget)
		predT = self.clf.forward_from_latent(latentT)
		
		if stage == 'train':
			discrepancy_loss = self.discLoss(latentT, latentS)
			m_loss_clf = self.clfLoss(predS, labSource)
			p_loss_clf = self.clDist(latentS, labSource)
			m_loss_AE = self.recLoss(dataTarget, rec)
			
			#clf_loss = m_loss_clf + self.hparams.alphaS * p_loss_clf + self.hparams.betaS * discrepancy_loss
			clf_loss = m_loss_clf + self.hparams.alphaS * p_loss_clf
			
			AE_loss = m_loss_AE + self.hparams.alphaT * discrepancy_loss
			
			metrics = {f'{stage}_m_loss_clf': m_loss_clf.detach(),
			           f'{stage}_m_loss_AE': m_loss_AE.detach(),
			           f'{stage}loss_disc': discrepancy_loss.detach(),
			           f'{stage}loss_clf': clf_loss.detach(),
			           f'{stage}loss_AE': AE_loss.detach()}
			
		elif stage =='val':
			#accSource = accuracy_score(labSource.detach().cpu(), np.argmax(predS.detach().cpu(), axis=1))
			#accTarget = accuracy_score(labTarget.detach().cpu(), np.argmax(predT.detach().cpu(), axis=1))
			discrepancy_loss = self.discLoss(latentT, latentS)
			m_loss_clf = self.clfLoss(predS, labSource)
			p_loss_clf = self.clDist(latentS, labSource)
			m_loss_AE = self.recLoss(dataTarget, rec)
			
			#clf_loss = m_loss_clf + self.hparams.alphaS * p_loss_clf + self.hparams.betaS * discrepancy_loss
			clf_loss = m_loss_clf + self.hparams.alphaS * p_loss_clf
			AE_loss = m_loss_AE + self.hparams.alphaT * discrepancy_loss
			metrics = {f'{stage}_m_loss_clf': m_loss_clf.detach(),
			           f'{stage}_m_loss_AE': m_loss_AE.detach(),
			           f'{stage}loss_disc': discrepancy_loss.detach(),
			           f'{stage}loss_clf': clf_loss.detach(),
			           f'{stage}loss_AE': AE_loss.detach()}
			# f'{stage}_acc_source': accSource,
			# f'{stage}_acc_target': accTarget,
		elif stage =='test':
			accSource = accuracy_score(labSource.cpu().numpy(), np.argmax(predS.cpu().numpy(), axis=1))
			accTarget = accuracy_score(labTarget.cpu().numpy(), np.argmax(predT.cpu().numpy(), axis=1))
			
			metrics ={'test_acc_source':accSource,
			          'all_acc_target':accTarget}
		return metrics

			
	def training_step(self, batch, batch_idx, optimizer_idx):
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'],source['label'].long()
		dataTarget = target['data']
		
		# we can put the data in GPU to process but with 'no_grad' pytorch way?
		# dataSource = dataSource.to(self.device, dtype=torch.float)
		# dataTarget = dataTarget.to(self.device, dtype=torch.float)
		# labSource = labSource.to(self.device, dtype=torch.long)
		# labTarget = labTarget.to(self.device, dtype=torch.long)
		
		if optimizer_idx == 0:
			latentS, predSource = self.clf(dataSource)  # call forward method
			m_loss = self.clfLoss(predSource, labSource)
			p_loss = self.clDist(latentS, labSource)
			
			#latentT, decoded = self.AE.forward(dataTarget)
			#discrepancy = self.discLoss(latentT, latentS)
			#loss = m_loss + self.hparams.alphaS * p_loss + self.hparams.betaS * discrepancy
			loss = m_loss + self.hparams.alphaS * p_loss
		elif optimizer_idx == 1:
			latentT, decoded = self.AE.forward(dataTarget)
			m_loss = self.recLoss(dataTarget, decoded)
			latentS, predSource = self.clf(dataSource)
			p_loss = self.discLoss(latentT, latentS)
			loss = m_loss +  self.hparams.alphaT * p_loss
		
		tqdm_dict = {f"{self.modelName[optimizer_idx]}_loss": loss}
		metrics = self._shared_eval_step(batch,stage = 'train')
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": metrics})
		return output
	
	def training_epoch_end(self,output):
		metrics = {}
		opt0 = [i['log'] for i in output[0]]
		opt1=[i['log'] for i in output[1]]
		
		keys_ = opt0[0].keys()
		for k in keys_:
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt0 ] + [i[k] for i in opt1]))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)



	def set_requires_grad(model, requires_grad=True):
		for param in self.clf.parameters():
			param.requires_grad = requires_grad
		for param in self.AE.parameters():
			param.requires_grad = requires_grad
			


	def validation_step(self, batch, batch_idx):
		#with torch.no_grad():
		metrics = self._shared_eval_step(batch,stage = 'val')

		return metrics
	
	def validation_epoch_end(self, out):
		keys_ = out[0].keys()
		metrics = {}
		for k in keys_:
			val = [i[k] for i in out]
			metrics[k] = torch.mean(torch.stack(val))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch,stage = 'test')
		for k,v in metrics.items():
			self.log(k,v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics

	def predict(self,dm_source,dm_target,getProb = False):
		with torch.no_grad():
			latentSource = []
			latentTarget = []
			predSource =[]
			predTarget = []
			trueSource = []
			trueTarget = []
			for source in dm_source.test_dataloader():
				dataSource, labS = source['data'], source['label'].long()
				l, pdS = self.clf(dataSource)
				latentSource.append(l.cpu().numpy())
				predSource.append(np.argmax(pdS.cpu().numpy(),axis = 1))
				trueSource.append(labS.cpu().numpy())
				
			for target in dm_target.dataloader():

				dataTarget,labT = target['data'],target['label'].long()

				l, rec = self.AE(dataTarget)
				latentTarget.append(l.cpu().numpy())
				pdT = self.clf.forward_from_latent(l)
				predTarget.append(np.argmax(pdT.cpu().numpy(),axis = 1))
				trueTarget.append(labT.cpu().numpy())

		predictions = {}
		predictions['latentSource'] = np.concatenate(latentSource,axis =0)
		predictions['predSource'] = np.concatenate(predSource,axis =0)
		predictions['trueSource'] = np.concatenate(trueSource,axis =0)
		predictions['latentTarget'] = np.concatenate(latentTarget,axis =0)
		predictions['predTarget'] = np.concatenate(predTarget,axis =0)
		predictions['trueTarget'] = np.concatenate(trueTarget,axis =0)
		return predictions


	def configure_optimizers(self):
		opt_clf = torch.optim.Adam(self.clf.parameters(), lr=self.hparams.lr_source)
		opt_AE = torch.optim.Adam(self.AE.parameters(), lr=self.hparams.lr_target)
		lr_sch_clf = StepLR(opt_clf, step_size=20, gamma=0.5)
		lr_sch_AE = StepLR(opt_AE, step_size=5, gamma=0.5)
		return [opt_clf, opt_AE], [lr_sch_clf,lr_sch_AE]
	
	# Alternating schedule for optimizer steps (e.g. GANs)
	def optimizer_step(
			self,
			epoch,
			batch_idx,
			optimizer,
			optimizer_idx,
			optimizer_closure,
			on_tpu=False,
			using_native_amp=False,
			using_lbfgs=False,
	):
		# update generator every step
		if epoch == self.hparams.max_eph_S:
			self.train_clf = False
			# for p in self.clf.parameters():
			# 	p.requires_grad = False
			# for ae in self.AE.parameters():
			# 	ae.requires_grad = True

		if optimizer_idx == 0:
			if self.train_clf:
				optimizer.step(closure=optimizer_closure)
			else:
				optimizer_closure()
		# update discriminator every 2 steps
		if optimizer_idx == 1:
			if not self.train_clf:
				optimizer.step(closure=optimizer_closure)
			else:
				optimizer_closure()

	def train_dataloader(self):
		return [self.dm_source.train_dataloader(),
		        self.dm_target.train_dataloader()]
	
	def test_dataloader(self):
		loaders = [self.dm_source.test_dataloader(),
		        self.dm_target.dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders
	
	def val_dataloader(self):
		loaders = [self.dm_source.val_dataloader(),
		        self.dm_target.val_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders

	def setDatasets(self,dm_source,dm_target):
		self.dm_source = dm_source
		self.dm_target = dm_target