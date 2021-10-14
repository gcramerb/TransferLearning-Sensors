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
	
	def __init__(self,n_class, hyp=None,n_sensor = 2):
		super(classifier, self).__init__()
		self.n_class = n_class
		self._name = 'clf'
		if hyp:
			self.conv_window = hyp['conv_window']
			self.conv_window2 = hyp['conv_window2']
			self.pooling_window_1 = hyp['pooling_window_1']
			self.same_pad = pooling_window_1
			self.pooling_window_2 = hyp['pooling_window_2']
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['encDim']
			
		else:
			self.conv_window = (5, 3)
			self.conv_window2 = (25, 3)
			self.pooling_window_1 = (2, 1)
			self.pooling_window_2 = (5, 1)
			self.n_filters = (4,8, 16, 32,64)
			self.encoded_dim = 50
		self.n_sensors = 2
			
		
	@property
	def name(self):
		return self._name
	
	def build(self):

		## encoder layers ##
		self.encoderSensor1 = nn.Sequential(
			nn.Conv2d(in_channels=1, kernel_size=self.conv_window, out_channels=self.n_filters[0], padding='same'),
			nn.BatchNorm2d(self.n_filters[0]),
			nn.ReLU(),
			nn.MaxPool2d(self.pooling_window_1)
		)

		self.encoderSensor2 = nn.Sequential(
			nn.Conv2d(in_channels=1, kernel_size=self.conv_window2, out_channels=self.n_filters[0], padding='same'),
			nn.BatchNorm2d(self.n_filters[0]),
			nn.ReLU(),
			nn.MaxPool2d(self.pooling_window_1)
		)
			
		self.mergedSameSensor = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[1], kernel_size=self.conv_window, out_channels=self.n_filters[2],
			          padding='same'),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.SELU(),
			nn.MaxPool2d(self.pooling_window_2)
		)
		
		self.mergedSensors = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[3], kernel_size=self.conv_window, out_channels=self.n_filters[3],
			          padding='same'),
			nn.BatchNorm2d(self.n_filters[3]),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(480, self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.SELU()
		)
		
		## decoder layers ##
		self.discrimination = nn.Sequential(
			nn.Linear(self.encoded_dim,self.n_class ),
			nn.Softmax(dim = 1)
		)
	
	def forward(self, X):
		AccEncoded1 = self.encoderSensor1(X[:, :, :, 0:3])
		GyrEncoded1 = self.encoderSensor1(X[:, :, :, 3:6])
		
		AccEncoded2 = self.encoderSensor2(X[:, :, :, 0:3])
		GyrEncoded2 = self.encoderSensor2(X[:, :, :, 3:6])
		
		acc =torch.cat([AccEncoded1, AccEncoded2], 1)
		gyr = torch.cat([GyrEncoded1, GyrEncoded2], 1)
		
		acc= self.mergedSameSensor(acc)
		gyr = self.mergedSameSensor(gyr)
		merged = torch.cat([acc, gyr], 1)
		encoded = self.mergedSensors(merged)
		pred = self.discrimination(encoded)
		return encoded, pred


