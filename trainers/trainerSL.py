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
			lossParams: dict = None,
			trashold: float = 0.75,
			save_path: str = None,
			class_weight: torch.tensor = None,
			**kwargs
	):
		super().__init__()
		self.ps = []
		self.trh =trashold
		self.batch_size = model_hyp['bs']
		self.n_classes = 4
		self.datasetTarget = "Dsads"
		
		self.save_hyperparameters()
		self.hparams.alpha = trainParams['alpha']
		self.hparams.penalty = trainParams['discrepancy']
		self.hparams.weight_decay = trainParams['weight_decay']
		self.hparams.dropout_rate = model_hyp['dropout_rate']
		self.hparams.lr = trainParams['lr']
		self.hparams.input_shape = model_hyp['input_shape']
		
		
	def create_model(self):
		self.clf = classifier(hyp=self.hparams.model_hyp,
		                  input_shape=self.hparams.input_shape,
		                      n_classes = self.n_classes)
		self.clf.build()
		self.clfLoss = nn.CrossEntropyLoss(weight=self.hparams.class_weight)
		
		# if self.hparams.penalty == 'mmd':
		# 	self.discLoss = MMDLoss()
		# elif self.hparams.penalty == 'ot':
		# 	self.discLoss = OTLoss(hyp=lossParams)
		# elif self.hparams.penalty == 'skn':
		# 	self.discLoss = SinkhornDistance(eps=1e-3, max_iter=200)
		# elif self.hparams.penalty == 'coral':
		# 	self.discLoss = CORAL()
		# else:
		# 	raise ValueError('specify a valid discrepancy loss!')


	def load_params(self, save_path, file):
		PATH = os.path.join(save_path, file + '_feature_extractorSL')
		self.clf.Encoder.load_state_dict(torch.load(PATH))
		PATH = os.path.join(save_path, file + '_discriminatorSL')
		self.clf.discrimination.load_state_dict(torch.load(PATH))
		for param in self.clf.parameters():
			param.requires_grad = True
			
	
	def save_params(self,save_path,file):
		path = os.path.join(save_path,file + '_feature_extractorSL')
		torch.save(self.clf.Encoder.state_dict(), path)
		path = os.path.join(save_path,file + '_discriminatorSL')
		torch.save(self.clf.discrimination.state_dict(), path)

	def generate_pseudoLab(self,path):
		with torch.no_grad():
			labT_ps = []
			dataT_ps = []

			for target in self.dm_target.train_dataloader():
				dataTarget= target['data']
				lat_,probs = self.clf(dataTarget)
				probs = probs.cpu().numpy()
				idx = np.where(probs.max(axis = 0)>self.trh)[0]
				labT_ps.append(np.argmax(probs[idx], axis=1))
				dataT_ps.append(dataTarget[idx])
			dataT = np.concatenate(dataT_ps, axis=0)
			labT = np.concatenate(labT_ps, axis=0)
		path_file = os.path.join(path,f'{self.datasetTarget}_pseudo_labels')
		if dataT.shape[1] ==2:
			dataT = np.concatenate([dataT[:,[0],:,:],dataT[:,[1],:,:]],axis = -1)
		np.savez(path_file,Xsl = dataT,ysl = labT)

	def compute_loss(self, batch):
		log = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		latentS,pred = self.clf(dataSource)
		
		loss = self.clfLoss(pred, labSource)
		
		# dataTarget = target['data']
		# latentT,_ = self.clf(dataTarget)
		# discrepancy = self.discLoss(latentT, latentS)
		# loss = loss + self.hparams.alpha * discrepancy

		logs = {}
		logs['loss'] = loss

		return loss,logs
	
	def _shared_eval_step(self, batch, stage='val'):
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		dataTarget, labTarget = target['data'], target['label'].long()

		latS,pred_S = self.clf(dataSource)
		latT,pred_T = self.clf(dataTarget)

		metrics = {}
		yhat_S = np.argmax(pred_S.detach().cpu().numpy(), axis=1)
		yhat_T = np.argmax(pred_T.detach().cpu().numpy(), axis=1)

		acc_S = accuracy_score(labSource.cpu().numpy(), yhat_S)
		acc_T = accuracy_score(labTarget.cpu().numpy(), yhat_T)
		
		metrics = {f'{stage}_acc_S': acc_S}
		metrics[f'{stage}_acc_T'] = acc_T

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
		return None
	
	def getPredict(self):
		"""

		:return:
		"""
		with torch.no_grad():
			source = self._predict(self.dm_source.test_dataloader(), 'Source')
			target = self._predict(self.dm_target.test_dataloader(), 'Target')
		
		out = {**source, **target}
		return out
	def get_final_metrics(self):
		result = {}
		predictions = self.getPredict()
		result['acc_source_test'] = accuracy_score(predictions['trueSource'], predictions['predSource'])
		result['acc_target_all'] = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
		result['cm_source'] = confusion_matrix(predictions['trueSource'], predictions['predSource'])
		result['cm_target'] = confusion_matrix(predictions['trueTarget'], predictions['predTarget'])
		return result
	
	def configure_optimizers(self):
		opt = torch.optim.Adam(self.clf.parameters(),
		                                    lr=self.hparams.lr,
		                                    weight_decay=self.hparams.weight_decay)

		
		#lr_sch = StepLR(opt_FE, step_size=20, gamma=0.5)
		return {"optimizer":opt}


	def _predict(self, dataloader,domain):
		latent = []
		probs = []
		y_hat = []
		true = []
		for data in dataloader:
			X, y = data['data'], data['label'].long()
			l = self.clf.getLatent(X)
			pred = self.clf.forward_from_latent(l).cpu().numpy()
			latent.append(l.cpu().numpy())
			probs.append(pred)
			y_hat.append(np.argmax(pred, axis=1))
			true.append(y.cpu().numpy())
		predictions = {}
		predictions['latent' +domain ] = np.concatenate(latent, axis=0)
		predictions['pred' + domain] = np.concatenate(y_hat, axis=0)
		predictions['true' + domain] = np.concatenate(true, axis=0)
		predictions['prob'  + domain] = np.concatenate(probs, axis=0)
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
		self.batch_size = dm_source.batch_size
		self.dm_source = dm_source
		self.dm_target = dm_target
		self.n_classes = dm_target.n_classes
		self.datasetTarget = dm_target.datasetName
