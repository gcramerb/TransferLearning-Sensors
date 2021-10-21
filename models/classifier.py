import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

from .customLosses import MMDLoss,OTLoss


# define the NN architecture
class classifier(nn.Module):
	"""

	"""
	
	def __init__(self,n_class, hyp=None):
		super(classifier, self).__init__()
		self.n_class = n_class
		self._name = 'clf'
		if hyp:
			self.conv_window = hyp['conv_window']
			self.conv_window2 = hyp['conv_window2']
			self.pooling_window_1 = hyp['pooling_window_1']
			self.same_pad = pooling_window_1
			self.pooling_2 = hyp['pooling_2']
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['encDim']
			
		else:
			self.conv_dim = [(5, 3),(25, 3)]
			self.pooling_1 = (2, 1)
			self.pooling_2 = (5, 1)
			self.n_filters = (4,8, 16, 32,64)
			self.encoded_dim = 50
		self.n_win = 2
		self.CNN1 = nn.ModuleList([])
			
		
	@property
	def name(self):
		return self._name
	
	def build(self):

		## encoder layers ##
		for i in range(self.n_win):
			self.CNN1.append(nn.Sequential(
				nn.Conv2d(in_channels=1, kernel_size=self.conv_dim[i],
				          out_channels=self.n_filters[i], padding='same',bias= False),
				nn.BatchNorm2d(self.n_filters[i]),
				nn.ReLU(),
				nn.MaxPool2d(self.pooling_1)))


			
		self.CNN2 = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[0] + self.n_filters[1] , kernel_size=self.conv_dim[0], out_channels=self.n_filters[2],
			          padding='same',bias = False),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.SELU(),
			nn.MaxPool2d(self.pooling_2)
		)
		
		self.DenseLayer = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[3], kernel_size=self.conv_dim[0], out_channels=self.n_filters[3],
			          padding='same',bias = False),
			nn.BatchNorm2d(self.n_filters[3]),
			nn.ReLU(),
			nn.Dropout(p=0.05, inplace=False),
			nn.Flatten(),
			nn.Linear(480, self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.SELU(),
			nn.Dropout(p=0.4, inplace=False)
		)
		
		## decoder layers ##
		self.discrimination = nn.Sequential(
			nn.Linear(self.encoded_dim,self.n_class ),
			nn.Softmax(dim = 1)
		)
	
	def forward(self, X):
		AccEncoded = []
		GyrEncoded = []
		for layer in self.CNN1:
			AccEncoded.append(layer(X[:, :, :, 0:3]))
			GyrEncoded.append(layer(X[:, :, :, 3:6]))

		acc =torch.cat(AccEncoded, 1)
		gyr = torch.cat(GyrEncoded, 1)
		
		acc= self.CNN2(acc)
		gyr = self.CNN2(gyr)
		merged = torch.cat([acc, gyr], 1)
		encoded = self.DenseLayer(merged)
		
		pred = self.discrimination(encoded)
		return encoded, pred




