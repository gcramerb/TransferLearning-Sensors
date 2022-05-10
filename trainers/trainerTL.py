import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse,glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier
from models.blocks import Encoder, discriminator, domainClf
from models.customLosses import MMDLoss, OTLoss, classDistance, SinkhornDistance, CORAL
from models.pseudoLabSelection import simplest_SLselec

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that trainers basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""
#TODO: implementar que cada modelo treine mais de uma epoca..

class TLmodel(LightningModule):
	
	def __init__(
			self,
			trainParams: dict = None,
			model_hyp: dict = None,
			lossParams: dict = None,
			save_path: str = None,
			penalty:str = 'ot',
			class_weight: torch.tensor = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.hparams.alpha = trainParams['alpha']
		self.hparams.penalty = trainParams['discrepancy']
		self.hparams.weight_decay = trainParams['weight_decay']
		self.hparams.dropout_rate = model_hyp['dropout_rate']
		self.hparams.lr_fe = trainParams['lr']
		self.hparams.lr_disc = trainParams['lr']
		self.hparams.input_shape = model_hyp['input_shape']
		
	def create_model(self):

		self.FE = Encoder(hyp=self.hparams.model_hyp,
			              input_shape=self.hparams.input_shape)
		self.FE.build()
		
		self.Disc  =  discriminator(dropout_rate = self.hparams.dropout_rate,
		                                 encoded_dim = self.hparams.model_hyp['enc_dim'],
		                                 n_classes = self.hparams.n_classes)
		self.Disc.build()
		
		self.clfLoss = nn.CrossEntropyLoss(weight=self.hparams.class_weight)

		if self.hparams.penalty == 'mmd':
			self.discLoss = MMDLoss()
		elif self.hparams.penalty == 'ot':
			#self.discLoss = OTLoss(hyp=lossParams)
			self.discLoss = OTLoss()
		elif self.hparams.penalty == 'skn':
			self.discLoss = SinkhornDistance(eps=1e-3, max_iter=200)
		elif self.hparams.penalty == 'coral':
			self.discLoss = CORAL()
		else:
			raise ValueError('specify a valid discrepancy loss!')


	def load_params(self, save_path, file):
		PATH = os.path.join(save_path, file + '_feature_extractor')
		self.FE.load_state_dict(torch.load(PATH))
		PATH = os.path.join(save_path, file + '_discriminator')
		self.Disc.load_state_dict(torch.load(PATH))
		for param in self.FE.parameters():
			param.requires_grad = True
		for param in self.Disc.parameters():
			param.requires_grad = True

	def save_params(self,save_path,file):
		path = os.path.join(save_path,file + '_feature_extractor')
		torch.save(self.FE.state_dict(), path)
		path = os.path.join(save_path,file + '_discriminator')
		torch.save(self.Disc.state_dict(), path)


	def compute_loss(self, batch,optimizer_idx):
		log = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		
		latentS= self.FE(dataSource)
		predS = self.Disc(latentS)
		
		loss = self.clfLoss(predS, labSource)
		if optimizer_idx ==0: # updating FE
			dataTarget = target['data']
			latentT = self.FE(dataTarget)
			discrepancy = self.discLoss(latentT, latentS)
			loss = loss + self.hparams.alpha * discrepancy

		logs = {}
		logs['loss'] = loss.detach()

		return loss,logs
	
	def _shared_eval_step(self, batch,optimizer_idx,stage='val'):
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()

		latS = self.FE(dataSource)
		pred_S = self.Disc(latS)
		
		latT = self.FE(dataTarget)
		pred_T = self.Disc(latT)
		metrics = {}
		yhat_S = np.argmax(pred_S.detach().cpu().numpy(), axis=1)
		yhat_T = np.argmax(pred_T.detach().cpu().numpy(), axis=1)

		acc_S = accuracy_score(labSource.cpu().numpy(), yhat_S)
		acc_T = accuracy_score(labTarget.cpu().numpy(), yhat_T)
		
		metrics = {f'{stage}_acc_S': acc_S}
		metrics[f'{stage}_acc_T'] = acc_T

		if stage == 'val':
			_, logs = self.compute_loss(batch,optimizer_idx)
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
		if isinstance(output[0], list):
			opt = [i['log'] for i in output[0]]
		else:
			opt = [i['log'] for i in output]
		
		for k in opt[0].keys():
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt]))
		
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return None
	
	def validation_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch, stage='val',optimizer_idx = 0)
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
		return None
	
	def configure_optimizers(self):
		opt_list = []
		opt_list.append(torch.optim.Adam(self.FE.parameters(),
		                                 lr=self.hparams.lr_fe,
		                                 weight_decay=self.hparams.weight_decay))
		
		opt_list.append(torch.optim.Adam(self.Disc.parameters(), lr=self.hparams.lr_disc))
		
		# lr_sch = StepLR(opt_FE, step_size=20, gamma=0.5)
		return opt_list, []
	
	def getPredict(self,domain = 'Target'):
		"""
		:return:
		"""
		
		with torch.no_grad():
			if domain =='Source':
				out = self._predict(self.dm_source.test_dataloader(), 'Source')
			elif domain =='Target':
				out = self._predict(self.dm_target.test_dataloader(), 'Target')
			else:
				raise ValueError('You must specify a valid domain!')
		return out
	def get_final_metrics(self):
		result = {}
		pred_source = self.getPredict(domain = 'Source')
		pred_target = self.getPredict(domain='Target')
		predictions = {**pred_source,**pred_target}

		result['acc_source_all'] = accuracy_score(predictions['trueSource'], predictions['predSource'])
		result['acc_target_all'] = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
		result['cm_source'] = confusion_matrix(predictions['trueSource'], predictions['predSource'])
		result['cm_target'] = confusion_matrix(predictions['trueTarget'], predictions['predTarget'])
		return result
	

	def _predict(self, dataloader,domain):
		latent = []
		probs = []
		y_hat = []
		true = []
		data_ori = []
		for data in dataloader:
			X, y = data['data'], data['label'].long()
			lat = self.FE(X)
			pred = self.Disc(lat).cpu().numpy()
			y_hat.append(np.argmax(pred, axis=1))
			latent.append(lat.cpu().numpy())
			probs.append(pred)
			true.append(y.cpu().numpy())
			data_ori.append(X)
		predictions = {}
		predictions['latent' +domain ] = np.concatenate(latent, axis=0)
		predictions['pred' + domain] = np.concatenate(y_hat, axis=0)
		predictions['true' + domain] = np.concatenate(true, axis=0)
		predictions['prob'  + domain] = np.concatenate(probs, axis=0)
		predictions['data' + domain] = np.concatenate(data_ori, axis=0)
		return predictions
	
	def train_dataloader(self):
		loaders = [self.dm_source.train_dataloader(),
		        self.dm_target.train_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders

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
		self.n_classes = dm_target.n_classes
		self.datasetTarget = dm_target.datasetName
