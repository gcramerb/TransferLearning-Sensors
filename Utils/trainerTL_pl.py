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


"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""
class TLmodel(LightningModule):
	
	def __init__(
			self,
			lr: float = 0.002,
			batch_size: list = [64,256],
			n_classes: int = 6,
			alphaS: float = 1.0,
			alphaT: float = 1.0,
			penalty: str = 'mmd',
			data_shape: tuple = (1,50,6),
			modelHyp: dict = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()

        # networks
		self.clf = classifier(6, modelName = 'clf1',hyp =self.hparams.modelHyp,inputShape = self.hparams.data_shape)
		self.AE = ConvAutoencoder(self.hparams.modelHyp)
		self.clf.build()
		self.AE.build()
		
		# from torchsummary import summary
		# summary(self.AE.to('cuda'), (1,50,6))
		
		#SET THE losses:
		self.recLoss = torch.nn.MSELoss()
		self.clfLoss = torch.nn.CrossEntropyLoss()
		if self.hparams.penalty == 'mmd':
			self.discLoss = MMDLoss()
		elif self.hparams.penalty == 'ot':
			self.discLoss = OTLoss()
		else:
			raise ValueError('specify a valid discrepancy loss!')
		
		self.clDist = classDistance()
		self.optNames = ['Classifier','Reconstructior']
		

	def forward(self, X):
		return self.clf(X)

	def _shared_eval_step(self, batch,optimizer_idx,stage ='train'):
		source, target = batch['source'], batch['target']
		dataSource, labSource = source
		dataTarget, labTarget = target
		
		# TODO: o labSource está errado!
		
		# we can put the data in GPU to process but with 'no_grad' pytorch way?
		dataSource = dataSource.to(self.device, dtype=torch.float)
		dataTarget = dataTarget.to(self.device, dtype=torch.float)
		labSource = labSource.to(self.device, dtype=torch.long)
		labTarget = labTarget.to(self.device, dtype=torch.long)

		if optimizer_idx ==0:
			latent, predSource = self.clf(dataSource) #call forward method
			m_loss = self.clfLoss(predSource, labSource)
			p_loss = self.clDist(latent, labSource)
			loss = m_loss + self.hparams.alphaS * p_loss
		elif optimizer_idx ==1:
			latentT, decoded = self.AE.forward(dataTarget)
			m_loss = self.recLoss(dataTarget, decoded)
			latentS,predSource = self.clf(dataSource)
			p_loss = self.discLoss(latentT,latentS)
			loss = m_loss + self.hparams.alphaT * p_loss
		else:
			raise ValueError(f"Optimizer number {optimizer_idx} not defined !!!")

		if stage =='val' or stage=='test':
			
			_, predTarget = self.clf(dataTarget)
			accSource = accuracy_score(labSource.cpu().numpy(), np.argmax(predSource.cpu().numpy(), axis=1))
			accTarget = accuracy_score(labTarget.cpu().numpy(), np.argmax(predTarget.cpu().numpy(), axis=1))
			loss = loss.item()
			metrics = {f"loss_{self.optNames[optimizer_idx]}": loss,
			           'accSource': accSource, 'accTarget': accTarget}
			return metrics
		return loss
			
	def training_step(self, batch, batch_idx, optimizer_idx):
		loss = self._shared_eval_step(batch=batch,optimizer_idx=optimizer_idx)
		tqdm_dict = {f"{self.optNames[optimizer_idx]}_loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output

	def set_requires_grad(model, requires_grad=True):
		for param in self.clf.parameters():
			param.requires_grad = requires_grad
		for param in self.AE.parameters():
			param.requires_grad = requires_grad
			
	def configure_optimizers(self):
		lr = self.hparams.lr
		opt_clf = torch.optim.Adam(self.clf.parameters(), lr=lr)
		opt_AE = torch.optim.Adam(self.AE.parameters(), lr=lr)
		return [opt_clf, opt_AE], []

	def validation_step(self, batch, batch_idx):

		metrics = self._shared_eval_step(batch,  optimizer_idx = 1, stage = 'val')
		# self.logger.experiment.log_dict('1',metrics,'val_metrics.txt')
		self.log('val_loss', metrics[f"loss_{self.optNames[1]}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('accValSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('accValTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics