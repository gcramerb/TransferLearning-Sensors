import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy


# define the NN architecture
class classifier(nn.Module):
	"""

	"""
	
	def __init__(self,n_class, hyp=None):
		super(classifier, self).__init__()
		self.n_class = n_class
		if hyp:
			self.conv_window = hyp['conv_window']
			self.pooling_window_1 = hyp['pooling_window_1']
			self.same_pad = pooling_window_1
			self.pooling_window_2 = hyp['pooling_window_2']
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['encDim']
		else:
			self.conv_window = (5, 3)
			self.pooling_window_1 = (2, 1)
			self.pooling_window_2 = (5, 1)
			self.n_filters = (8, 16, 32)
			self.encoded_dim = 50
		
	
	def build(self):

		## encoder layers ##
		self.encoderSensor = nn.Sequential(
			nn.Conv2d(in_channels=1, kernel_size=self.conv_window, out_channels=self.n_filters[0], padding='same'),
			nn.BatchNorm2d(self.n_filters[0]),
			nn.ReLU(),
			nn.MaxPool2d(self.pooling_window_1),
			nn.Conv2d(in_channels=self.n_filters[0], kernel_size=self.conv_window, out_channels=self.n_filters[1],
			          padding='same'),
			nn.BatchNorm2d(self.n_filters[1]),
			nn.ReLU(),
			nn.MaxPool2d(self.pooling_window_2)
		)
		self.mergedSensors = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[2], kernel_size=self.conv_window, out_channels=self.n_filters[2],
			          padding='same'),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(480, self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.ReLU()
		)
		
		## decoder layers ##
		self.discrimination = nn.Sequential(
			nn.Linear(self.encoded_dim,self.n_class ),
			nn.Softmax(dim = 1)
		)
	
	def forward(self, X):
		AccEncoded = self.encoderSensor(X[:, :, :, 0:3])
		GyrEncoded = self.encoderSensor(X[:, :, :, 3:6])
		encoded = self.mergedSensors(torch.cat([AccEncoded, GyrEncoded], 1))
		pred = self.discrimination(encoded)
		return encoded, pred


