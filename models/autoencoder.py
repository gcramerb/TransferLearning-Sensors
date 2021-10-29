import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

from .blocks import Encoder

# define the NN architecture
class ConvAutoencoder(nn.Module):
	"""
	
	"""
	def __init__(self, hyp=None):
		super(ConvAutoencoder, self).__init__()
		self._name = 'AE'
		self.Encoder = Encoder(hyp)

	
	@property
	def name(self):
		return self._name
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
		self.Encoder.build()



	def forward(self, X):
		AccEncoded = self.encoderSensor(X[:,:,:,0:3])
		GyrEncoded = self.encoderSensor(X[:, :, :, 3:6])
		encoded = self.mergedSensors(torch.cat([AccEncoded,GyrEncoded] ,1))
		decoded = self.decoder(encoded)
		return encoded,decoded
	

