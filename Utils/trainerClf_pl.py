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


class networkLight(LightningModule):
	def __init__(
			self,
			lr: float = 0.002,
			n_classes: int = 6,
			alpha: float = 1.0,
			inputShape:tuple = (1,50,6),
			FeName: str = 'fe1',
			modelHyp: dict = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.model = classifier(6,self.hparams.FeName, self.hparams.modelHyp,inputShape = inputShape)
		self.model.build()
		self.m_loss = torch.nn.CrossEntropyLoss()
		self.p_loss = classDistance()


	def forward(self, X):
		return self.model(X)

	def set_requires_grad(model, requires_grad=True):
		for param in self.model.parameters():
			param.requires_grad = requires_grad
	
	def training_step(self, batch, batch_idx):
		# opt = self.optimizers()
		data,  label = batch['data'], batch['label']
		latent, pred = self.model(data)
		label = label.long()  # why need this?
		loss = self.m_loss(pred, label) + self.hparams.alpha * self.p_loss(latent, label)
		tqdm_dict = {"train_loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output

	def validation_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		for k,v in metrics.items():
			self.log(f'val_{k}',v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics

	def test_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		self.log('test_acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics

	def _shared_eval_step(self, batch, batch_idx):
		data, label = batch['data'], batch['label'].long()
		latent, pred = self.model(data)

		m_loss = self.m_loss(pred, label)
		p_loss = self.p_loss(latent,label)
		loss = m_loss  + self.hparams.alpha * p_loss

		acc = accuracy_score(label.cpu().numpy(), np.argmax(pred.cpu().numpy(), axis=1))
		loss = loss.item()

		metrics = {"loss": loss,
		           'm_loss':m_loss,
		           'p_loss': p_loss,
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

	def configure_optimizers(self):
		opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
		lr_scheduler = StepLR(opt, step_size=self.hparams.step_size, gamma=0.5)
		#return {"optimizerClf": opt, "lr_scheduler": self.schedulerClf}
		return [opt], [lr_scheduler]

