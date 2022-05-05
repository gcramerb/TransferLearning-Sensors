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
from models.customLosses import MMDLoss,OTLoss, classDistance
#import geomloss


from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict


class ClfModel(LightningModule):
	def __init__(
			self,
			lr: float = 0.002,
			n_classes: int = 6,
			alpha: float = 1.0,
			input_shape:tuple = (1,50,6),
			step_size = 10,
			model_hyp: dict = None,
			weight_decay: float = 0.0,
			class_weight:torch.tensor = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.model = classifier(n_classes,
		                        hyp = self.hparams.model_hyp,
		                        input_shape=self.hparams.input_shape)
		self.model.build()
		self.m_loss = torch.nn.CrossEntropyLoss(weight = self.hparams.class_weight)
		# self.p_loss = classDistance()
	
	def save_params(self,save_path,file):
		path = os.path.join(save_path,file + '_feature_extractor')
		torch.save(self.model.Encoder.state_dict(), path)
		path = os.path.join(save_path,file + '_discriminator')
		torch.save(self.model.discrimination.state_dict(), path)
	
	def load_params(self,save_path,file):
		path = os.path.join(save_path,file + '_feature_extractor')
		self.model.Encoder.load_state_dict(torch.load(path))
		path = os.path.join(save_path,file + '_discriminator')
		self.model.discrimination.load_state_dict(torch.load(path))

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
		# p_loss = self.p_loss(latent,label)
		# loss = m_loss  + self.hparams.alpha * p_loss
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
			probs = []
			X = []
			for batch in dataLoaderTest:
				data, label = batch['data'], batch['label']
				l, pdS = self.model(data)
				latent.append(l.cpu().numpy())
				probs.append(pdS.cpu().numpy())
				pred.append(np.argmax(pdS.cpu().numpy(), axis=1))
				true.append(label.cpu().numpy())
				X.append(data.cpu().numpy())
		predictions = {}
		predictions['latent'] = np.concatenate(latent, axis=0)
		predictions['pred'] = np.concatenate(pred, axis=0)
		predictions['probs'] = np.concatenate(probs,axis =0)
		predictions['true'] = np.concatenate(true, axis=0)
		predictions['data'] = np.concatenate(X,axis =0)
		
		return predictions
	
	def get_all_metrics(self,dm):
		stage = 'test'
		metric = {}
		dataLoaders = dm.test_dataloader()
		predictions = self.predict(dataLoaders)
		metric[f'{stage}_acc'] =  accuracy_score(predictions['true'],predictions['pred'])
		metric[f'{stage}_cm'] = confusion_matrix(predictions['true'],predictions['pred'])
		return metric

	def configure_optimizers(self):
		opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
		if self.hparams.step_size:
			lr_scheduler = StepLR(opt, step_size=self.hparams.step_size, gamma=0.5)
		#return {"optimizerClf": opt, "lr_scheduler": self.schedulerClf}
			return [opt], [lr_scheduler]
		else:
			return [opt]
	

