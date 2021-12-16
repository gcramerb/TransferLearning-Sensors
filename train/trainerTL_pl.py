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
from models.blocks import Encoder2, Encoder1,domainClf
from models.customLosses import MMDLoss,OTLoss, classDistance,SinkhornDistance,CORAL
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
			lr_gan: float = 0.00005,
			n_classes: int = 6,
			alphaS: float = 1.0,
			betaS: float = 0.5,
			alphaT: float = 1.0,
			penalty: str = 'mmd',
			data_shape: tuple = (1,50,6),
			modelHyp: dict = None,
			FeName: str = 'fe2',
			weight_decay:float = 0.0,
			feat_eng: str = 'sym',
			epch_rate: int = 8,
			lossParams: dict = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		

        # networks
		self.clf = classifier(n_classes = self.hparams.n_classes,
		                      FeName=self.hparams.FeName,
		                      hyp=self.hparams.modelHyp,
		                      inputShape=self.hparams.data_shape)
		
		# self.AE = ConvAutoencoder(FeName = self.hparams.FeName,
		#                           hyp = self.hparams.modelHyp,
		#                           inputShape = self.hparams.data_shape)
		self.AE = Encoder2(hyp = self.hparams.modelHyp,
		                         inputShape = self.hparams.data_shape)
		
		self.domainClf = domainClf(self.hparams.modelHyp['encDim'])
		
		self.clf.build()
		self.AE.build()
		self.domainClf.build()
		
		self.test_metrics = []
		self.train_clf = True

		# from torchsummary import summary
		# summary(self.AE.to('cuda'), (1,50,6))
		
		#SET the losses:
		#self.recLoss = torch.nn.MSELoss()
		self.GanLoss = nn.BCELoss()
		#self.GanLoss = torch.nn.SoftMarginLoss()
		self.clfLoss = torch.nn.CrossEntropyLoss()
		
		#self.clfLoss = torch.nn.NLLLoss()
		if self.hparams.penalty == 'mmd':
			self.discLoss = MMDLoss()
		elif self.hparams.penalty == 'ot':
			self.discLoss = OTLoss(hyp = lossParams)
		elif self.hparams.penalty =='skn':
			self.discLoss = SinkhornDistance(eps=1e-3, max_iter=200)
		elif self.hparams.penalty =='coral':
			self.discLoss = CORAL()
		else:
			raise ValueError('specify a valid discrepancy loss!')
		
		self.clDist = classDistance()
		self.modelName = ['Classifier', 'Reconstructior','Adiversarial']
		self.datasetName = ['Source','Target']
		

	def forward(self, X):
		return self.clf(X)
	
	def compute_loss(self,batch,model = 'clf'):
		log = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget = target['data']
		
		latentS, predSource = self.clf(dataSource)  # call forward method
		latentT = self.AE.forward(dataTarget)
		if model =='clf':
			m_loss = self.clfLoss(predSource, labSource)
			log[f'm_loss_clf'] = m_loss
			#p_loss = self.clDist(latentS, labSource)
			if self.hparams.feat_eng == 'sym':
				discrepancy = self.discLoss(latentT, latentS)
				ganLoss = self.get_GAN_loss(latentS, latentT)
				# loss = m_loss + self.hparams.alphaS * p_loss - self.hparams.betaS * ganLoss
				loss = m_loss + self.hparams.alphaS * discrepancy - self.hparams.betaS * ganLoss
			elif self.hparams.feat_eng == 'asym':
				loss = m_loss
			else:
				raise ValueError('wrong feat_eng value')
		if model =='AE':
			ganLoss = self.get_GAN_loss(latentS,latentT)
			# latentT, decoded = self.AE.forward(dataTarget)
			# m_loss = self.recLoss(dataTarget, decoded)
			
			discrepancy = self.discLoss(latentT, latentS)
			# predT = self.clf.forward_from_latent(latentT.detach())
			# clas_dist = self.clDist(latentT,np.argmax(predT.detach().cpu(),axis = 1))
			#loss = p_loss
			loss = discrepancy -1* self.hparams.alphaT * ganLoss
			log['discpy_loss'] = discrepancy
			log['GAN_loss'] = ganLoss
		if model =='GAN':
			loss = self.get_GAN_loss(latentS,latentT)
		return loss,log
	
	def get_GAN_loss(self,latentS,latentT):
		yS = torch.ones(latentS.shape[0], 1)
		yT = torch.zeros(latentT.shape[0], 1)
		x = torch.cat([latentS, latentT])
		y = torch.cat([yS, yT])
		pred = self.domainClf(x)
		return self.GanLoss(pred, y.to(self.device))
	
	def _shared_eval_step(self, batch,stage = 'val'):

		source,target = batch[0],batch[1]
		dataSource,labSource = source['data'],source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()
		latentS, predS = self.clf(dataSource)
		#latentT,rec = self.AE.forward(dataTarget)
		latentT = self.AE.forward(dataTarget)
		predT = self.clf.forward_from_latent(latentT)

		metrics = {}
		if stage =='test':
			yhatS = np.argmax(predS.detach().cpu().numpy(), axis=1)
			yhatT = np.argmax(predT.detach().cpu().numpy(), axis=1)
			accSource = accuracy_score(labSource.cpu().numpy(), yhatS)
			accTarget = accuracy_score(labTarget.cpu().numpy(), yhatT)
			metrics ={f'{stage}_acc_source':accSource,
			          f'{stage}_acc_target':accTarget}
		else:
			_,logs = self.compute_loss(batch,'clf')
			_, logs_ae = self.compute_loss(batch, 'AE')
			logs.update(logs_ae)
			for k,v in logs.items():
				metrics[stage + '_' + k] = v.detach()
		return metrics

	def training_step(self, batch, batch_idx, optimizer_idx):
		if optimizer_idx == 0:
			loss,_ = self.compute_loss(batch,'clf')
		elif optimizer_idx == 1:
			loss,_ = self.compute_loss(batch, 'AE')
		elif optimizer_idx == 2:
			loss,_ = self.compute_loss(batch, 'GAN')
			
		tqdm_dict = {f"{self.modelName[optimizer_idx]}_loss": loss}
		#metrics = self._shared_eval_step(batch,stage = 'train')
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	def training_epoch_end(self,output):
		metrics = {}
		opt0 = [i['log'] for i in output[0]]
		opt1=[i['log'] for i in output[1]]
		opt2 = [i['log'] for i in output[2]]

		for k in opt0[0].keys():
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt0 ] ))
		for k in opt1[0].keys():
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt1 ]))
		for k in opt2[0].keys():
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt2]))
		
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

	def get_final_metrics(self):
		result = { }
		predictions = self.predict(self.dm_source,self.dm_target)
		result['acc_source_test'] = accuracy_score(predictions['trueSource'],predictions['predSource'])
		result['acc_target_all'] = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
		return result



	def configure_optimizers(self):
		opt_clf = torch.optim.Adam(self.clf.parameters(),
		                           lr=self.hparams.lr_source,
		                           weight_decay =  self.hparams.weight_decay)
		opt_AE = torch.optim.Adam(self.AE.parameters(), lr=self.hparams.lr_target)
		opt_domain = torch.optim.RMSprop(self.domainClf.parameters(), lr=self.hparams.lr_gan)
		
		# lr_sch_clf = StepLR(opt_clf, step_size=20, gamma=0.5)
		# lr_sch_AE = StepLR(opt_AE, step_size=10, gamma=0.5)
		return [opt_clf, opt_AE,opt_domain]
		#return [opt_clf, opt_AE], [lr_sch_clf,lr_sch_AE]

	# def optimizer_step(
	# 		self,
	# 		epoch,
	# 		batch_idx,
	# 		optimizer,
	# 		optimizer_idx,
	# 		optimizer_closure,
	# 		on_tpu=False,
	# 		using_native_amp=False,
	# 		using_lbfgs=False,
	# ):
	# 	# update generator every step
	# 	if (epoch+1) % self.hparams.epch_rate ==0 or epoch ==0:
	# 		self.train_clf = True
	# 		self.train_AE = False
	# 	else:
	# 		self.train_clf = False
	# 		self.train_AE = True
	# 	if optimizer_idx == 0:
	# 		if self.train_clf:
	# 			optimizer.step(closure=optimizer_closure)
	# 		else:
	# 			optimizer_closure()
	# 	# update discriminator every 2 steps
	# 	if optimizer_idx == 1:
	# 		if self.train_AE:
	# 			optimizer.step(closure=optimizer_closure)
	# 		else:
	# 			optimizer_closure()
	def predict(self, dm_source, dm_target):
		with torch.no_grad():
			latentSource = []
			latentTarget = []
			predSource = []
			predTarget = []
			trueSource = []
			trueTarget = []
			probTarget = []
			for source in dm_source.test_dataloader():
				dataSource, labS = source['data'], source['label'].long()
				l, pdS = self.clf(dataSource)
				latentSource.append(l.cpu().numpy())
				predSource.append(np.argmax(pdS.cpu().numpy(), axis=1))
				trueSource.append(labS.cpu().numpy())
			
			for target in dm_target.dataloader():
				dataTarget, labT = target['data'], target['label'].long()
				
				l = self.AE(dataTarget)
				latentTarget.append(l.cpu().numpy())
				pdT = self.clf.forward_from_latent(l)
				probs = pdT.cpu().numpy()
				probTarget.append(probs)
				predTarget.append(np.argmax(probs, axis=1))
				trueTarget.append(labT.cpu().numpy())
		
		predictions = {}
		predictions['latentSource'] = np.concatenate(latentSource, axis=0)
		predictions['predSource'] = np.concatenate(predSource, axis=0)
		predictions['trueSource'] = np.concatenate(trueSource, axis=0)
		predictions['latentTarget'] = np.concatenate(latentTarget, axis=0)
		predictions['predTarget'] = np.concatenate(predTarget, axis=0)
		predictions['trueTarget'] = np.concatenate(trueTarget, axis=0)
		predictions['probTarget'] = np.concatenate(probTarget, axis=0)
		return predictions
	
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
