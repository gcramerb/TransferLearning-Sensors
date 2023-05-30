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
from models.customLosses import CenterLoss

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict


class ClfModel(LightningModule):
	def __init__(
			self,
			trainParams: dict = None,
			n_classes: int = -1,
			class_weight: torch.tensor = None,
			oneHotLabel: bool = False,
			mixup: bool = False,
			**kwargs
	):
		super().__init__()
		self.hparams.alpha = trainParams['alpha']
		self.hparams.weight_decay = trainParams['weight_decay']
		self.hparams.encoded_dim = trainParams['enc_dim']
		self.hparams.dropout_rate = trainParams['dropout_rate']
		self.hparams.lr = trainParams['lr']
		self.hparams.input_shape = trainParams['input_shape']
		self.hparams.n_filters = trainParams['n_filters']
		self.hparams.kernel_dim = trainParams['kernel_dim']
		self.hparams.oneHotLabel = oneHotLabel
		self.hparams.mixup = mixup
		self.save_hyperparameters()
		self.classDist = None
	
	def create_model(self, setCenterLoss = True):
		self.model = classifier(n_classes=self.hparams.n_classes,
		                        dropout_rate=self.hparams.dropout_rate,
		                        encoded_dim=self.hparams.encoded_dim,
		                        input_shape=self.hparams.input_shape,
		                        n_filters=self.hparams.n_filters,
		                        kernel_dim=self.hparams.kernel_dim
		                        )
		self.model.create_model()
		self.clfLoss = torch.nn.CrossEntropyLoss(weight=self.hparams.class_weight)
		if setCenterLoss:
			self.classDist = CenterLoss(num_classes=self.hparams.n_classes, feat_dim=self.hparams.encoded_dim, use_gpu=True)
	
	def save_params(self, save_path, file):
		path = os.path.join(save_path, file + '_feature_extractor')
		torch.save(self.model.FE.state_dict(), path)
		path = os.path.join(save_path, file + '_discriminator')
		torch.save(self.model.Disc.state_dict(), path)
	
	def load_params(self, save_path, file):
		path = os.path.join(save_path, file + '_feature_extractor')
		self.model.FE.load_state_dict(torch.load(path))
		path = os.path.join(save_path, file + '_discriminator')
		self.model.Disc.load_state_dict(torch.load(path))
		for param in self.FE.parameters():
			param.requires_grad = False
		for param in self.Disc.parameters():
			param.requires_grad = False
	def load_featureExtractor(self, save_path, file):
		PATH = os.path.join(save_path, file + '_feature_extractor')
		weights = torch.load(PATH)
		self.model.FE.load_state_dict(weights)
		for param in self.model.FE.parameters():
			param.requires_grad = False
		for param in self.model.Disc.parameters():
			param.requires_grad = True
	def load_paramsFL(self, save_path, file):
		"""
		Just load the first layer. if the weights were loaded, we freezes these initial layers

		:param save_path:
		:param file:
		:return:
		"""
		path = os.path.join(save_path, file + '_feature_extractor')
		pretrained_dict = torch.load(path)
		
		# TODO: testar se esta conjelando as camadas corretas.
		i = 0
		processed_dict = {}
		model_dict = self.model.FE.state_dict()  # new model keys
		
		for k in model_dict.keys():
			
			decomposed_key = k.split(".")
			if ("model" in decomposed_key):
				pretrained_key = ".".join(decomposed_key[1:])
				processed_dict[k] = pretrained_dict[
					pretrained_key]  # Here we are creating the new state dict to make our new model able to load the pretrained parameters without the head.
			i = i + 1
			if i > 1:
				break
		self.model.FE.load_state_dict(processed_dict, strict=False)
	
	def forward(self, X):
		return self.model.forward(X)
	
	def set_requires_grad(model, requires_grad=True):
		for param in self.model.parameters():
			param.requires_grad = requires_grad
	
	def mixup(self, latent, label, weight):
		indices = torch.randperm(latent.size(0), device=latent.device, dtype=torch.long)
		perm_latent = latent[indices]
		perm_label = label[indices]
		return latent.mul(weight).add(perm_latent, alpha=1 - weight), label.mul(weight).add(perm_label,
		                                                                                    alpha=1 - weight)
	
	def training_step(self, batch, batch_idx):
		# opt = self.optimizers()
		data, label = batch['data'], batch['label'].long()
		latent = self.model.getLatent(data)
		pred = self.model.forward_from_latent(latent)
		loss = self.clfLoss(pred, label)
		if self.classDist is not None:
			classDistence = self.classDist(latent, label)
			loss += self.hparams.alpha * classDistence
		tqdm_dict = {"train_loss": loss.detach()}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		self.log("training_loss", loss, batch_size=self.batch_size)
		return output
	
	def training_epoch_end(self, output):
		metrics = {}
		opt = [i['log'] for i in output]
		
		keys_ = opt[0].keys()
		for k in keys_:
			metrics[k] = torch.mean(torch.stack([i[k] for i in opt]))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	def validation_step(self, batch, batch_idx, dataloader_idx):
		return batch
	
	def validation_epoch_end(self, out):
		Ypred = []
		Ytrue = []
		for batch in out[0]:
			data, label = batch['data'], batch['label'].long()
			latent, pred = self.model(data)
			Ypred.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
			Ytrue.append(label.cpu().numpy())
		
		Ytrue = np.concatenate(Ytrue, axis=0)
		Ypred = np.concatenate(Ypred, axis=0)
		metrics = {}
		if len(Ytrue.shape)>1:
			Ytrue = np.argmax(Ytrue, axis=1)
		metrics['valAcc (ps)'] = accuracy_score(Ytrue, Ypred)
		# metrics['len ps'] = len(Ytrue)
		Ypred = []
		Ytrue = []
		for batch in out[1]:
			data, label = batch['data'], batch['label'].long()
			latent, pred = self.model(data)
			Ypred.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
			Ytrue.append(label.cpu().numpy())
		Ytrue = np.concatenate(Ytrue, axis=0)
		Ypred = np.concatenate(Ypred, axis=0)
		if len(Ytrue.shape)>1:
			Ytrue = np.argmax(Ytrue, axis=1)
		metrics['valAcc (target)'] = accuracy_score(Ytrue, Ypred)
		# metrics['len target'] = len(Ytrue)
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	def predict(self, dataloader):
		latent = []
		pred = []
		true = []
		probs = []
		X = []
		for batch in dataloader:
			data, label = batch['data'], batch['label']
			l, pdS = self.model(data)
			latent.append(l.detach().cpu().numpy())
			probs.append(pdS.detach().cpu().numpy())
			pred.append(np.argmax(pdS.detach().cpu().numpy(), axis=1))
			true.append(label.cpu().numpy())
			X.append(data.detach().cpu().numpy())
		predictions = {}
		predictions['latent'] = np.concatenate(latent, axis=0)
		predictions['pred'] = np.concatenate(pred, axis=0)
		predictions['probs'] = np.concatenate(probs, axis=0)
		predictions['true'] = np.concatenate(true, axis=0)
		predictions['data'] = np.concatenate(X, axis=0)
		return predictions
	
	def configure_optimizers(self):
		opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		# if self.hparams.step_size:
		# 	lr_scheduler = StepLR(opt, step_size=self.hparams.step_size, gamma=0.5)
		# 	return [opt], [lr_scheduler]
		# else:
		return opt
	
	def train_dataloader(self):
		return self.dm.train_dataloader()
	
	def test_dataloader(self):
		if self.scnd_dm is not None:
			return [self.dm.test_dataloader(), self.scnd_dm.test_dataloader()]
		else:
			return self.dm.test_dataloader()
	
	def val_dataloader(self):
		if self.scnd_dm is not None:
			return [self.dm.val_dataloader(), self.scnd_dm.val_dataloader()]
		else:
			return [self.dm.val_dataloader(), self.dm.val_dataloader()]
	
	def setDatasets(self, dm, secondDataModule=None):
		self.dm = dm
		self.n_classes = dm.n_classes
		self.batch_size = dm.batch_size
		self.scnd_dm = secondDataModule
