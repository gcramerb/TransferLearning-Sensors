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
from dataProcessing.create_dataset import crossDataset, targetDataset, getData


from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict


class networkLight(LightningModule):
	def __init__(
			self,
			lr: float = 0.0002,
			batch_size: int = 128,
			n_classes: int = 6,
			alpha: float = 1.0,
			penalty: str = 'mmd',
			modelHyp: dict = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.model = classifier(6, self.hparams.modelHyp)
		self.model.build()
		self.m_loss , self.p_loss = self.myLoss(self.hparams.penalty)
		#self.automatic_optimization = True
		
	
	def myLoss(self, penalty):

		if penalty == 'mmd':
			 penalty_loss =  MMDLoss()
		elif penalty == "ot":
			penalty_loss = OTLoss()
		elif penalty == 'ClDist':
			penalty_loss = classDistance()
		else:
			penalty_loss = None
		return torch.nn.CrossEntropyLoss(),penalty_loss
	
	def forward(self, X):
		return self.model(X)
	
	# def adversarial_loss(self, y_hat, y):
	#     return F.binary_cross_entropy(y_hat, y)
	def set_requires_grad(model, requires_grad=True):
		for param in self.model.parameters():
			param.requires_grad = requires_grad
	
	def training_step(self, batch, batch_idx):

		# opt = self.optimizers()
		data, domain, label = batch['data'], batch['domain'], batch['label']

		latent, pred = self(data)

		sourceIdx = np.where(domain.cpu().numpy() == 0)[0]
		true = label[sourceIdx]
		pred = pred[sourceIdx]

		true = true.long()  # why need this?
		loss = self.m_loss(pred, true) + self.hparams.alpha * self.p_loss(latent, domain,true)

		#
		# opt.zero_grad()
		# self.manual_backward(loss)
		# opt.step()
		#
		tqdm_dict = {"loss": loss}

		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	def training_step_end(self, training_step_outputs):
		metrics = training_step_outputs['log']
		loss = metrics['loss'].item()
		#print(loss)
		#print(self.model.CNN2[0].weight.grad)
		# self.logger.experiment.log_metric(self.logger.run_id, key='training_loss',
		#                                   value=loss)
		self.log('training_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

		print('training_loss:  ',loss)
	def validation_step(self, batch, batch_idx):

		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']

		# self.logger.experiment.log_dict('1',metrics,'val_metrics.txt')
		self.log('val_loss', metrics['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('accValSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('accValTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

		# self.logger.experiment.log_metric(self.logger.run_id,key= 'val_loss', value= metrics['loss'])
		# self.logger.experiment.log_metric(self.logger.run_id,'accValSource', metrics['accSource'])
		# self.logger.experiment.log_metric(self.logger.run_id,'accValTarget', metrics['accTarget'])
		# print('val_loss: ',  metrics['loss'],' ','accValSource: ',
		#                                    metrics['accSource'],' ','accValTarget: ',metrics['accTarget'])
		return metrics

	def test_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		# self.log_dict(metrics)
		# self.logger.experiment.log_metric(self.logger.run_id,'accTestSource', metrics['accSource'])
		# self.logger.experiment.log_metric(self.logger.run_id,'accTestTarget', metrics['accTarget'])
		self.log('accTestSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('accTestTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics

	def _shared_eval_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		latent, pred = self(data)
		#pred = self(data)

		sourceIdx = np.where(domain.cpu().numpy() == 0)[0]
		targetIdx = np.where(domain.cpu().numpy() != 0)[0]
		trueSource = label[sourceIdx]
		predSource = pred[sourceIdx]
		trueTarget = label[targetIdx]
		predTarget = pred[targetIdx]
		outputs = (trueSource, predSource, trueTarget, predTarget)
		# return outputs
		trueSource, predSource, trueTarget, predTarget = outputs

		trueSource = trueSource.long()  # why need this?
		loss = self.m_loss(predSource, trueSource)  + self.hparams.alpha * self.p_loss(latent,domain,trueSource)
		#loss = self.m_loss(predSource, trueSource)
		accSource = accuracy_score(trueSource.cpu().numpy(), np.argmax(predSource.cpu().numpy(), axis=1))
		accTarget = accuracy_score(trueTarget.cpu().numpy(), np.argmax(predTarget.cpu().numpy(), axis=1))
		loss = loss.item()

		metrics = {"loss": loss,
		           'accSource': accSource, 'accTarget': accTarget}

		tqdm_dict = metrics
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		}
		return result

	def predict(self, dataTest):
		pass
		# print('predict')
		# latent, domain, pred = [], [], []
		# data = np.concatenate([x['data'] for x in dataTest])
		# d = [x['domain'] for x in dataTest]
		# label = [x['label'] for x in dataTest]
		#
		# #l, p = self.model.forward(torch.tensor(np.expand_dims(data, axis=1)))
		# pred = np.argmax(p.numpy(), axis=1)
		# return l.numpy(), d, pred.tolist(), label

	def configure_optimizers(self):
		opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
		lr_scheduler = StepLR(opt, step_size=30, gamma=0.5)
		#return {"optimizer": opt, "lr_scheduler": self.scheduler}
		return [opt], [lr_scheduler]

	def on_epoch_end(self):
		print('epoch_end')
		pass

	# def automatic_optimization(self):
	# 	print('auto opt')
	# 	"""If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``."""
	# 	return False
	
	# def manual_backward(self, loss, optimizer):
	# 	print('manual back')
	# 	loss.backward()
		#optimizer.step()
	# def manual_backward(self, loss) -> None:
	# 	# make sure we're using manual opt
	# 	self._verify_is_manual_optimization("manual_backward")
	#
	# def configure_optimizers(self):
	# 	return optim.Adam(self.model.parameters(), lr=self.hparams.lr)
	
	def on_epoch_end(self):
		pass




