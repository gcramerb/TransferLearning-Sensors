import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
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
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()

        # networks
		self.clf = classifier(6, FeName = FeName,hyp =self.hparams.modelHyp,inputShape = self.hparams.data_shape)
		self.AE = ConvAutoencoder(FeName = FeName,hyp = self.hparams.modelHyp)
		self.clf.build()
		self.AE.build()
		self.test_metrics = []
		
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
		self.optNames = ['Classifier','Reconstructior']
		

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
		
		source, target = batch['source'], batch['target']
		dataSource, labSource = source
		dataTarget, labTarget = target
		
		labTarget = labTarget.long()
		labSource = labSource.long()
		
		latentT,rec = self.AE.forward(dataTarget)
		predTarget = self.clf.forward_from_latent(latentT)
		latentS, predSource = self.clf(dataSource)
		if stage == 'val':
			discrepLoss = self.discLoss(latentT, latentS)
			clfLoss =  self.clfLoss(predSource, labSource) + self.hparams.alphaS * self.clDist(latentS, labSource) + self.hparams.betaS * discrepLoss
			AEloss = self.recLoss(dataTarget, rec) + self.hparams.alphaT * discrepLoss
			metrics = self._get_metrics(labSource, predSource, labTarget, predTarget,AEloss,clfLoss)
		elif stage =='test':
			metrics = self._get_metrics(labSource, predSource, labTarget, predTarget)
		return metrics

			
	def training_step(self, batch, batch_idx, optimizer_idx):
		source, target = batch['source'], batch['target']
		dataSource, labSource = source
		dataTarget, labTarget = target
		
		# we can put the data in GPU to process but with 'no_grad' pytorch way?
		# dataSource = dataSource.to(self.device, dtype=torch.float)
		# dataTarget = dataTarget.to(self.device, dtype=torch.float)
		# labSource = labSource.to(self.device, dtype=torch.long)
		# labTarget = labTarget.to(self.device, dtype=torch.long)
		
		if optimizer_idx == 0:
			latentS, predSource = self.clf(dataSource)  # call forward method
			m_loss = self.clfLoss(predSource, labSource.long())
			p_loss = self.clDist(latentS, labSource)
			
			latentT, decoded = self.AE.forward(dataTarget)
			discrepancy = self.discLoss(latentT, latentS)
			loss = m_loss + self.hparams.alphaS * p_loss + self.hparams.betaS * discrepancy
		elif optimizer_idx == 1:
			latentT, decoded = self.AE.forward(dataTarget)
			m_loss = self.recLoss(dataTarget, decoded)
			latentS, predSource = self.clf(dataSource)
			p_loss = self.discLoss(latentT, latentS)
			loss = m_loss + self.hparams.alphaT * p_loss
		tqdm_dict = {f"{self.optNames[optimizer_idx]}_loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	def training_epoch_end(self,output):
		train_lossClf = []
		train_lossAE = []
		if len(output) ==2:
			for clfMetrics in output[0]:
				train_lossClf.append(clfMetrics['loss'].item())
			for AEmetrics in output[1]:
				train_lossAE.append(AEmetrics['loss'].item())
			self.log('train_loss_classifier',np.mean(train_lossClf), on_step=False, on_epoch=True, prog_bar=True, logger=True)
			self.log('train_loss_AE', np.mean(train_lossAE), on_step=False, on_epoch=True, prog_bar=True,
			         logger=True)
	def validation_epoch_end(self,out):
		out = out[0]
		for k,v in out.items():
			self.log('val_' + k,v, on_step=False, on_epoch=True, prog_bar=True,logger=True)
		
	def set_requires_grad(model, requires_grad=True):
		for param in self.clf.parameters():
			param.requires_grad = requires_grad
		for param in self.AE.parameters():
			param.requires_grad = requires_grad
			


	def validation_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch,stage = 'val')
		# self.logger.experiment.log_dict('1',metrics,'val_metrics.txt')
		self.log(f"val_loss_AE", metrics['AEloss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log(f"val_loss_clf", metrics['clfLoss'], on_step=False, on_epoch=True, prog_bar=True,
		         logger=True)
		#self.log('accValSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		#self.log('accValTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics
	

	def test_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch,stage = 'test')

		self.log('accTestSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('accTestTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics
	
	def predict(self,dataLoaderTest):
		with torch.no_grad():
			latentSource = []
			latentTarget = []
			predSource =[]
			predTarget = []
			trueSource = []
			trueTarget = []
			for batch in dataLoaderTest:
				source, target = batch['source'], batch['target']
				dataSource, labS = source
				dataTarget, labT = target
				l, pdS = self.clf(dataSource)
				latentSource.append(l.cpu().numpy())
				predSource.append(np.argmax(pdS.cpu().numpy(),axis = 1))
				trueSource.append(labS.cpu().numpy())
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
		

	def on_test_end(self):
		accSource = np.mean([a['accSource'] for a in self.test_metrics])
		accTarget = np.mean([a['accTarget'] for a in self.test_metrics])
		print(accSource,accTarget)
	def configure_optimizers(self):
		opt_clf = torch.optim.Adam(self.clf.parameters(), lr=self.hparams.lr_source)
		opt_AE = torch.optim.Adam(self.AE.parameters(), lr=self.hparams.lr_target)
		return [opt_clf, opt_AE], []
