import torch
import torch.nn as nn
from torch.nn.functional import  cross_entropy
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
# seed = 19
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier,classifierTest
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss, OTLoss, classDistance
# import geomloss



from dataProcessing.create_dataset import crossDataset, targetDataset, getData
from dataProcessing.dataModule import CrossDatasetModule

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

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
		self.model = classifierTest(7, self.hparams.modelHyp)
		self.model.build()
		
	def forward(self, x):
		# use forward for inference/predictions
		embedding = self.model(x)
		return embedding
	
	def training_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		pred = self(data)
		# sourceIdx = np.where(domain.numpy() == 0)[0]
		# label = label[sourceIdx]
		# pred = pred[sourceIdx]
		loss = cross_entropy(pred, label.long())
		self.log("train_loss", loss, on_epoch=True)
		#print(loss)
		print(self.model.CNN1.weight.grad)
		return loss
	
	def validation_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		pred = self(data)
		# sourceIdx = np.where(domain.numpy() == 0)[0]
		# label = label[sourceIdx]
		# pred = pred[sourceIdx]
		loss = cross_entropy(pred, label.long())
		
		self.log("valid_loss", loss, on_step=True)
	
	def test_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		pred = self(data)
		# sourceIdx = np.where(domain.numpy() == 0)[0]
		# label = label[sourceIdx]
		# pred = pred[sourceIdx]
		loss = cross_entropy(pred, label.long())
		self.log("test_loss", loss)
	
	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		return self(data)
	
	def configure_optimizers(self):
		# self.hparams available because we called self.save_hyperparameters()
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
	
	
def cli_main():
	inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	dm = CrossDatasetModule(data_dir=inPath,source = 'Pamap2')
	dm.setup()
	model = networkLight(penalty=None, alpha=None, lr=0.001)
	trainer = Trainer(gpus = 1,check_val_every_n_epoch=10, max_epochs=10, progress_bar_refresh_rate=0)
	trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
	cli_main()