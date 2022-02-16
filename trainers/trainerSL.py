import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier
from models.blocks import Encoder, discriminator, domainClf
from models.customLosses import MMDLoss, OTLoss, classDistance, SinkhornDistance, CORAL
# import geomloss

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that trainers basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""
#TODO: implementar que cada modelo treine mais de uma epoca..

class SLmodel(LightningModule):
	
	def __init__(
			self,
			trainParams: dict = None,
			model_hyp: dict = None,
			n_classes: int = 6,
			lossParams: dict = None,
			save_path: str = None,
			class_weight: torch.tensor = None,
			**kwargs
	):
		super().__init__()
		self.ps_from_a = []
		self.ps_from_b = []
		self.trh = 0.75
		
		self.save_hyperparameters()
		self.hparams.alpha = trainParams['alpha']
		self.hparams.beta = trainParams['beta']
		self.hparams.penalty = trainParams['discrepancy']
		self.hparams.weight_decay = trainParams['weight_decay']
		self.hparams.dropout_rate = model_hyp['dropout_rate']
		self.hparams.lr = trainParams['lr']
		
		self.hparams.input_shape = model_hyp['input_shape']
		
		self.FE_a = Encoder(hyp=self.hparams.model_hyp,
		                  input_shape=self.hparams.input_shape)
		self.FE_a.build()
		self.FE_b = Encoder(hyp=self.hparams.model_hyp,
		                  input_shape=self.hparams.input_shape)
		self.FE_b.build()
		self.dicr_a = discriminator(dropout_rate=self.hparams.dropout_rate,
		                                encoded_dim=self.hparams.model_hyp['enc_dim'],
		                                n_classes=self.hparams.n_classes)
		self.dicr_a.build()
		self.dicr_b = discriminator(dropout_rate=self.hparams.dropout_rate,
		                                encoded_dim=self.hparams.model_hyp['enc_dim'],
		                                n_classes=self.hparams.n_classes)
		self.dicr_b.build()

		self.clfLoss = nn.CrossEntropyLoss(weight=self.hparams.class_weight)


	def load_params(self, save_path, file):
		PATH = os.path.join(save_path, file + '_feature_extractor')
		self.FE_a.load_state_dict(torch.load(PATH))
		PATH = os.path.join(save_path, file + '_discriminator')
		self.dicr_a.load_state_dict(torch.load(PATH))
		for param in self.FE_a.parameters():
			param.requires_grad = True
		for param in self.dicr_a.parameters():
			param.requires_grad = True

	def generate_pseudoLab(self,data,predLabels):
		return [np.argmax(l) for l in predLabels if max(l) > self.trh],  [d for d,l in zip(data,predLabels) if max(l) > self.trh]

	def compute_loss(self, batch,batch_idx, optmizer_idx):
		log = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget = target['data']
		
		if optmizer_idx = 0:

			if len(self.ps_from_b) > self.batch_size:
				idx_ = (batch_idx * self.batch_size)%(len(self.ps_from_b) - self.batch_size)
				dataSL,labSL = self.ps_from_b[idx_:idx_ + self.dm_source.batch_size]
			else:
				dataSL,labSL = self.ps_from_b
				
			X = torch.concatenate([dataSL,dataSource])
			y = torch.concatenate([labSL,labSource])
			
			latent = self.FE_a(X)
			pred = self.discr_a(latent)
			loss = self.clfLoss(pred,y)

			#now, get the pseudoLabels for the other model:
			latentT = self.FE_a(dataTarget)
			predT = self.discr_a(latentT)
			self.ps_from_a += self.generate_pseudoLab(latentT, dataTarget)
		
		if optmizer_idx = 1:
			# da para tirar essa parte do codigo se as variaveis forem dicionario e acessar pelo iptimizer_ix...

			if len(self.ps_from_a) > self.batch_size:
				idx_ = (batch_idx * self.batch_size) % (len(self.ps_from_a) - self.batch_size)
				dataSL, labSL = self.ps_from_a[idx_:idx_ + self.dm_source.batch_size]
			else:
				dataSL, labSL = self.ps_from_a
			
			X = torch.concatenate([dataSL, dataSource])
			y = torch.concatenate([labSL, labSource])
			
			latent = self.FE_a(X)
			pred = self.discr_a(latent)
			loss = self.clfLoss(pred, y)
			
			# now, get the pseudoLabels for the other model:
			latentT = self.FE_b(dataTarget)
			predT = self.discr_b(latentT)
			self.ps_from_b += self.generate_pseudoLab(latentT, dataTarget)

		return loss
	
	def _shared_eval_step(self, batch, stage='val'):
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()

		latent_aS = self.FE_a(dataSource)
		pred_aS = self.disc_a(latent_aS)
		latent_aT = self.FE_a(dataTarget)
		pred_aT = self.disc_a(latent_aT)
		
		latent_bS = self.FE_b(dataSource)
		pred_bS = self.disc_b(latent_bS)
		latent_bT = self.FE_b(dataTarget)
		pred_bT = self.disc_b(latent_bT)
		

		metrics = {}
		yhat_aS = np.argmax(pred_aS.detach().cpu().numpy(), axis=1)
		yhat_aT = np.argmax(pred_aT.detach().cpu().numpy(), axis=1)
		yhat_bS = np.argmax(pred_bS.detach().cpu().numpy(), axis=1)
		yhat_bT = np.argmax(pred_bT.detach().cpu().numpy(), axis=1)
		
		acc_aS = accuracy_score(labSource.cpu().numpy(), yhat_aS)
		acc_aT = accuracy_score(labTarget.cpu().numpy(), yhat_aT)
		acc_bS = accuracy_score(labSource.cpu().numpy(), yhat_bS)
		acc_bT = accuracy_score(labTarget.cpu().numpy(), yhat_bT)
		
		metrics = {f'{stage}_acc_aS': acc_aS}
		metrics[f'{stage}_acc_aT'] = acc_aT
		metrics[f'{stage}_acc_bS'] = acc_bS
		metrics[f'{stage}_acc_bT'] = acc_bT
		
		if stage == 'val':
			_, logs = self.compute_loss(batch, 0)
			for k, v in logs.items():
				metrics[stage + '_' + k] = v.detach()
		return metrics
	
	def training_step(self, batch, batch_idx, optimizer_idx):
		loss, log = self.compute_loss(batch, optimizer_idx)
		tqdm_dict = log
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	

	def get_final_metrics(self):
		result = {}
		predictions = self.predict()
		result['acc_source_test'] = accuracy_score(predictions['trueSource'], predictions['predSource'])
		result['acc_target_all'] = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
		result['cm_source'] = confusion_matrix(predictions['trueSource'], predictions['predSource'])
		result['cm_target'] = confusion_matrix(predictions['trueTarget'], predictions['predTarget'])
		return result
	
	def configure_optimizers(self):
		opt_list = []
		opt_list.append(torch.optim.Adam(self.FE.parameters(),
		                                    lr=self.hparams.lr,
		                                    weight_decay=self.hparams.weight_decay))
		opt_list.append(torch.optim.Adam(self.FE.parameters(),
		                                    lr=self.hparams.lr,
		                                    weight_decay=self.hparams.weight_decay))
		
		# lr_sch_FE = StepLR(opt_FE, step_size=20, gamma=0.5)
		return opt_list, []
	
	
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
				
				l = self.staticFE(dataSource) if self.hparams.feat_eng == 'asym' else self.FE(dataSource)
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
		self.batch_size = dm_source.batch_size
		self.dm_source = dm_source
		self.dm_target = dm_target
