import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, '../')

from models.classifier import classifier
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss, classDistance
#import geomloss


from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict


class networkLight(LightningModule):
	def __init__(
			self,
			lr: float = 0.002,
			n_classes: int = 6,
			alpha: float = 1.0,
			inputShape:tuple = (1,50,6),
			FeName: str = 'fe2',
			step_size = 10,
			modelHyp: dict = None,
			weight_decay: float = 0.0,
			class_weight:torch.tensor = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.model = classifier(n_classes, self.hparams.FeName, self.hparams.modelHyp, inputShape=inputShape)
		self.model.build()
		self.m_loss = torch.nn.CrossEntropyLoss(weight = self.hparams.class_weight)
		#self.p_loss = classDistance()
	
	def save_params(self,save_path,file):
		path = os.path.join(save_path,file + '_feature_extractor')
		torch.save(self.model.Encoder.state_dict(), path)
		path = os.path.join(save_path,file + '_discriminator')
		torch.save(self.model.discrimination.state_dict(), path)

	def forward(self, X):
		return self.model(X)

	def set_requires_grad(model, requires_grad=True):
		for param in self.model.parameters():
			param.requires_grad = requires_grad
	
	def training_step(self, batch, batch_idx):
		# opt = self.optimizers()
		data,  label = batch['data'], batch['label'].long()
		latent, pred = self.model(data)

		#loss = self.m_loss(pred, label) + self.hparams.alpha * self.p_loss(latent, label)
		loss =  self.m_loss(pred, label)
		tqdm_dict = {"train_loss": loss.detach()}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	def training_epoch_end(self, output):
		metrics = {}
		opt = [i['log'] for i in output]
		
		keys_ = opt[0].keys()
		for k in keys_:
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt] ))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	def validation_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		return metrics

	def validation_epoch_end(self, out):
		keys_ = out[0].keys()
		metrics = {}
		for k in keys_:
			val = [i[k] for i in out]
			if k =='acc':
				metrics['val_'+k] = np.mean(val)
			else:
				metrics['val_' + k] = torch.mean(torch.stack(val))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)


	def test_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		self.log('test_acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics

	def _shared_eval_step(self, batch, batch_idx):
		data, label = batch['data'], batch['label'].long()
		latent, pred = self.model(data)

		m_loss = self.m_loss(pred, label)
		#p_loss = self.p_loss(latent,label)
		#loss = m_loss  + self.hparams.alpha * p_loss
		loss = m_loss

		acc = accuracy_score(label.cpu().numpy(), np.argmax(pred.detach().cpu(), axis=1))

		metrics = {"loss": loss,
		           'm_loss':m_loss,
		           #'p_loss': p_loss,
		           'acc': acc}

		tqdm_dict = metrics
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		}
		return result

	def predict(self, dataLoaderTest):
		with torch.no_grad():
			latent = []
			pred = []
			true = []
			for batch in dataLoaderTest:
				data, label = batch['data'], batch['label']
				l, pdS = self.model(data)
				latent.append(l.cpu().numpy())
				pred.append(np.argmax(pdS.cpu().numpy(), axis=1))
				true.append(label.cpu().numpy())
		predictions = {}
		predictions['latent'] = np.concatenate(latent, axis=0)
		predictions['pred'] = np.concatenate(pred, axis=0)
		predictions['true'] = np.concatenate(true, axis=0)
		return predictions
	
	def get_all_metrics(self,dm):
		metric = {}
		dataLoaders = [dm.train_dataloader(),dm.val_dataloader(),dm.test_dataloader()]
		with torch.no_grad():
			for i,stage in enumerate(['train','val','test']):
				pred = []
				true = []
				cm = np.zeros([self.hparams.n_classes, self.hparams.n_classes])
				for batch in dataLoaders[i]:
					
					data, label = batch['data'], batch['label']
					l, pdS = self.model(data)
					true_batch =label.cpu().numpy()
					pred_batch = np.argmax(pdS.cpu().numpy(), axis=1)
					pred.append(pred_batch)
					true.append(true_batch)
					cm += confusion_matrix(true_batch,pred_batch)
				true = np.concatenate(true)
				pred= np.concatenate(pred)
				metric[f'{stage}_acc'] =  accuracy_score(true,pred)
				metric[f'{stage}_cm'] = cm
		return metric

	def configure_optimizers(self):
		opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
		lr_scheduler = StepLR(opt, step_size=self.hparams.step_size, gamma=0.5)
		#return {"optimizerClf": opt, "lr_scheduler": self.schedulerClf}
		return [opt], [lr_scheduler]

