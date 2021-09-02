import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

# define the NN architecture
class ConvAutoencoder(nn.Module):
	"""
	
	"""
	def __init__(self, hyp=None):
		super(ConvAutoencoder, self).__init__()
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
		class myReshape(nn.Module):
			def __init__(self, newShape):
				super(myReshape, self).__init__()
				self.shape = newShape
			def forward(self, x):
				bs = x.shape[0]
				newShape =tuple([bs]+ list(self.shape))
				return x.reshape(newShape)

		## encoder layers ##
		self.encoderSensor = nn.Sequential(
			nn.Conv2d(in_channels=1, kernel_size=self.conv_window, out_channels=self.n_filters[0], padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(self.pooling_window_1),
			nn.Conv2d(in_channels=self.n_filters[0], kernel_size=self.conv_window, out_channels=self.n_filters[1],padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(self.pooling_window_2)
		)
		self.mergedSensors = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[2], kernel_size=self.conv_window, out_channels=self.n_filters[2], padding='same'),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(480, self.encoded_dim),
			nn.ReLU()
		)
		
		## decoder layers ##
		self.decoder = nn.Sequential(
			nn.Linear(self.encoded_dim, 480),
			nn.ReLU(),
			myReshape((32,5,3)),
			nn.ConvTranspose2d(in_channels=self.n_filters[2], kernel_size=self.conv_window, out_channels=self.n_filters[1],stride = (5,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=self.n_filters[1], kernel_size=self.conv_window, out_channels=self.n_filters[0],stride=(2, 1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=self.n_filters[0], kernel_size=self.conv_window, out_channels=1,stride=(2, 1)),
			nn.ReLU()
		)

	def forward(self, X):

		AccEncoded = self.encoderSensor(X[:,:,:,0:3])
		GyrEncoded = self.encoderSensor(X[:, :, :, 3:6])
		encoded = self.mergedSensors(torch.cat([AccEncoded,GyrEncoded] ,1))
		decoded = self.decoder(encoded)
		return encoded,decoded


