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
from models.blocks import Encoder2, Encoder1,discriminator
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
			n_classes: int = 6,
			penalty: str = 'mmd',
			data_shape: tuple = (1, 50, 6),
			modelHyp: dict = None,
			FeName: str = 'fe2',
			weight_decay: float = 0.0,
			DropoutRate = 0.2,
			lossParams: dict = None,
			save_path: str = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		
		if FeName =='fe2':
			# load pre trined model?
			self.FE = Encoder2(hyp=self.hparams.modelHyp,
			                   inputShape=self.hparams.data_shape)
			

			self.staticFE =  Encoder2(hyp=self.hparams.modelHyp,
		                   inputShape=self.hparams.data_shape)

		self.staticDisc  =  discriminator(self.hparams.DropoutRate,
		                                 self.hparams.modelHyp['encDim'],
		                                 self.hparams.n_classes)
		self.FE.build()
		self.staticFE.build()
		self.staticDisc.build()

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


	def load_params(self,save_path):
		PATH = os.path.join(save_path, 'feature_extractor')
		self.FE.load_state_dict(torch.load(PATH))
		self.staticFE.load_state_dict(torch.load(PATH))
		for param in self.staticFE.parameters():
			param.requires_grad = False
		PATH = os.path.join(save_path, 'discriminator')
		self.staticDisc.load_state_dict(torch.load(PATH))
		
		for param in self.staticDisc.parameters():
			param.requires_grad = False

	def forward(self, X):
		return self.FE(X)
	
	def compute_loss(self, batch):
		log = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget = target['data']
		
		latentS = self.staticFE(dataSource)	# call forward method
		latentT = self.FE(dataTarget)
		discrepancy = self.discLoss(latentT, latentS)
		loss = discrepancy
		log['discpy_loss'] = discrepancy
		return loss,log

	def _shared_eval_step(self, batch, stage='val'):
		
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()
		
		latentS = self.staticFE(dataSource)
		predS = self.staticDisc(latentS)
		latentT = self.FE(dataTarget)
		predT = self.staticDisc(latentT)
		

		metrics = {}
		yhatS = np.argmax(predS.detach().cpu().numpy(), axis=1)
		yhatT = np.argmax(predT.detach().cpu().numpy(), axis=1)
		accSource = accuracy_score(labSource.cpu().numpy(), yhatS)
		accTarget = accuracy_score(labTarget.cpu().numpy(), yhatT)
		metrics = {f'{stage}_acc_source': accSource,
		           f'{stage}_acc_target': accTarget}
		if stage == 'val':
			_, logs = self.compute_loss(batch)
			for k, v in logs.items():
				metrics[stage + '_' + k] = v.detach()
		return metrics
	
	def training_step(self, batch, batch_idx):

		loss, log = self.compute_loss(batch)
		tqdm_dict = log
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	def training_epoch_end(self, output):
		metrics = {}
		opt = [i['log'] for i in output]

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
		return result
	
	def configure_optimizers(self):
		opt_FE = torch.optim.Adam(self.FE.parameters(),
		                           lr=self.hparams.lr,
		                           weight_decay=self.hparams.weight_decay)

		lr_sch_FE = StepLR(opt_FE, step_size=20, gamma=0.5)
		return [opt_FE],[lr_sch_FE]
	

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
				l = self.staticFE(dataSource)
				pdS = self.staticDisc(l)
				latentSource.append(l.cpu().numpy())
				predSource.append(np.argmax(pdS.cpu().numpy(), axis=1))
				trueSource.append(labS.cpu().numpy())
			
			for target in self.dm_target.dataloader():
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
		           self.dm_target.dataloader()]
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
