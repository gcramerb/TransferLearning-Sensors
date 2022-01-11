import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier
from models.autoencoder import ConvAutoencoder
from models.blocks import Encoder2, Encoder1,discriminator,domainClf
from models.customLosses import MMDLoss, OTLoss, classDistance, SinkhornDistance, CORAL
# import geomloss

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""


class FTmodel(LightningModule):
	
	def __init__(
			self,
			lr: float = 0.002,
			lr_gan:float = 0.0001,
			gan:bool = True,
			n_classes: int = 6,
			alpha: float = 1.0,
			beta: float = 0.75,
			penalty: str = 'ot',

			model_hyp: dict = None,
			weight_decay: float = 0.0,
			dropout_rate:float = 0.2,
			feat_eng:str = 'asym',
			lossParams: dict = None,
			save_path: str = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.hparams.input_shape = model_hyp['input_shape']
		
		if model_hyp['FE'] == 'fe2':
			# load pre trined model?
			self.FE = Encoder2(hyp=self.hparams.model_hyp,
			                   input_shape=self.hparams.input_shape)
			
		if self.hparams.feat_eng =='asym':
			self.staticFE =  Encoder2(hyp=self.hparams.model_hyp,
		                   input_shape=self.hparams.input_shape)
			self.staticFE.build()
		self.staticDisc  =  discriminator(self.hparams.dropout_rate,
		                                 self.hparams.model_hyp['enc_dim'],
		                                 self.hparams.n_classes)
		self.FE.build()
		self.staticDisc.build()

		## GAN:
		self.domainClf = domainClf(self.hparams.model_hyp['enc_dim'])
		self.domainClf.build()
		self.GanLoss = nn.BCELoss()
		
		self.clfLoss = nn.CrossEntropyLoss()

		if self.hparams.penalty == 'mmd':
			self.discLoss = MMDLoss()
		elif self.hparams.penalty == 'ot':
			self.discLoss = OTLoss(hyp=lossParams)
		elif self.hparams.penalty == 'skn':
			self.discLoss = SinkhornDistance(eps=1e-3, max_iter=200)
		elif self.hparams.penalty == 'coral':
			self.discLoss = CORAL()
		else:
			raise ValueError('specify a valid discrepancy loss!')


	def load_params(self,save_path,file):
		PATH = os.path.join(save_path, file + '_feature_extractor')
		self.FE.load_state_dict(torch.load(PATH))
		if self.hparams=='asym':
			self.staticFE.load_state_dict(torch.load(PATH))
			for param in self.staticFE.parameters():
				param.requires_grad = False
		PATH = os.path.join(save_path, file+'_discriminator')
		self.staticDisc.load_state_dict(torch.load(PATH))
		train = True if self.hparams.feat_eng =='sym' else False
		for param in self.staticDisc.parameters():
			param.requires_grad = train

	def forward(self, X):
		return self.FE(X)
	def get_GAN_loss(self,latentS,latentT):
		yS = torch.ones(latentS.shape[0], 1)
		yT = torch.zeros(latentT.shape[0], 1)
		x = torch.cat([latentS, latentT])
		y = torch.cat([yS, yT])
		pred = self.domainClf(x)
		return self.GanLoss(pred, y.to(self.device))
	
	def compute_loss(self, batch,optmizer_idx):
		log = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget = target['data']
		latentT = self.FE(dataTarget)
		
		if self.hparams.feat_eng =='asym':
			latentS = self.staticFE(dataSource)  # call forward method
			discrepancy = self.discLoss(latentT, latentS)
			m_loss = discrepancy
		else:
			latentS = self.FE(dataSource)
			predS = self.staticDisc(latentS)
			clf_loss = self.clfLoss(predS,labSource)
			log['clf_loss'] = clf_loss
			discrepancy = self.discLoss(latentT, latentS)
			m_loss = discrepancy + self.hparams.beta * clf_loss
			
		GAN_loss =  self.get_GAN_loss(latentS,latentT)
		if optmizer_idx == 0:
			loss = m_loss - 1*self.hparams.alpha*GAN_loss
		if optmizer_idx ==1:
			loss = GAN_loss
		if optmizer_idx == 2:
			loss = clf_loss
		log['discpy_loss'] = discrepancy
		log['GAN_loss'] = GAN_loss
		return loss,log

	def _shared_eval_step(self, batch, stage='val'):
		
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()
		if self.hparams.feat_eng =='asym':
			latentS = self.staticFE(dataSource)
		else:
			latentS = self.FE(dataSource)
		predS = self.staticDisc(latentS)
		latentT = self.FE(dataTarget)
		predT = self.staticDisc(latentT)
		metrics = {}
		yhatS = np.argmax(predS.detach().cpu().numpy(), axis=1)
		yhatT = np.argmax(predT.detach().cpu().numpy(), axis=1)
		accSource = accuracy_score(labSource.cpu().numpy(), yhatS)
		accTarget = accuracy_score(labTarget.cpu().numpy(), yhatT)
		metrics ={f'{stage}_acc_target': accTarget}
		metrics = {f'{stage}_acc_source': accSource}
		if stage == 'val':
			_, logs = self.compute_loss(batch,0)
			for k, v in logs.items():
				metrics[stage + '_' + k] = v.detach()
		return metrics
	
	def training_step(self, batch, batch_idx,optimizer_idx):
		loss, log = self.compute_loss(batch,optimizer_idx)
		tqdm_dict = log
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	def training_epoch_end(self, output):
		metrics = {}
		opt = [i['log'] for i in output[0]]

		for k in opt[0].keys():
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt]))

		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

	def validation_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch, stage='val')
		return metrics
	
	def validation_epoch_end(self, out):
		keys_ = out[0].keys()
		metrics = {}
		for k in keys_:
			val = [i[k] for i in out]
			if 'acc' in k:
				metrics[k] = np.mean(val)
			else:
				metrics[k] = torch.mean(torch.stack(val))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	def test_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch, stage='test')
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics
	
	def get_final_metrics(self):
		result = {}
		predictions = self.predict()
		result['acc_source_test'] = accuracy_score(predictions['trueSource'], predictions['predSource'])
		result['acc_target_all'] = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
		for k, v in result.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return result
	
	def configure_optimizers(self):
	
		opt_FE = torch.optim.RMSprop(self.FE.parameters(),
		                           lr=self.hparams.lr,
		                           weight_decay=self.hparams.weight_decay)
		opt_GAN= torch.optim.Adam(self.domainClf.parameters(), lr=self.hparams.lr_gan)
		if self.hparams.feat_eng == 'sym':
			opt_discrimin = torch.optim.Adam(self.staticDisc.parameters(), lr=self.hparams.lr_gan)
			return [opt_FE, opt_GAN,opt_discrimin], []

		#lr_sch_FE = StepLR(opt_FE, step_size=20, gamma=0.5)
		return [opt_FE,opt_GAN],[]
	

	def predict(self):
		with torch.no_grad():
			latentSource = []
			latentTarget = []
			
			predSource = []
			predTarget = []
			trueSource = []
			trueTarget = []
			probTarget = []
			for source in self.dm_source.test_dataloader():
				dataSource, labS = source['data'], source['label'].long()
				
				l = self.staticFE(dataSource) if self.hparams.feat_eng =='asym' else self.FE(dataSource)
				pdS = self.staticDisc(l)
				
				latentSource.append(l.cpu().numpy())
				predSource.append(np.argmax(pdS.cpu().numpy(), axis=1))
				trueSource.append(labS.cpu().numpy())
			
			for target in self.dm_target.train_dataloader():
				dataTarget, labT = target['data'], target['label'].long()
				l_tar = self.FE(dataTarget)
				latentTarget.append(l_tar.cpu().numpy())
				pdT = self.staticDisc(l_tar)
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
		           self.dm_target.test_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders
	
	def val_dataloader(self):
		loaders = [self.dm_source.val_dataloader(),
		           self.dm_target.val_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders
	
	def setDatasets(self, dm_source, dm_target):
		self.dm_source = dm_source
		self.dm_target = dm_target
