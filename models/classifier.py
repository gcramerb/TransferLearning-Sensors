import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

from .customLosses import MMDLoss, OTLoss
from .blocks import Encoder1,Encoder2

# define the NN architecture
class classifier(nn.Module):
	"""

	"""
	def __init__(self,
	             n_classes,
	             FeName = None,
	             hyp=None,
	             inputShape = (1,50,6)):
		super(classifier, self).__init__()
		self.n_classes = n_classes
		self._name = FeName
		if FeName =='fe1':
			self.Encoder = Encoder1(hyp=hyp,inputShape=inputShape)
		elif FeName == "fe2":
			self.Encoder = Encoder2(hyp=hyp,inputShape=inputShape)
		else:
			raise ValueError("Put a value model name!" )
	@property
	def name(self):
		return self._name
	def build(self):
		
		self.Encoder.build()
		self.discrimination = nn.Sequential(
			nn.Linear(self.Encoder.encoded_dim, self.n_classes),
			nn.Softmax(dim=1)
		)
		# from torchsummary import summary
		# summary(self.Encoder.to('cuda'), (2,50,3))

	def forward(self, X):
		encoded = self.Encoder.forward(X)
		pred = self.discrimination(encoded)
		return encoded, pred
	def forward_from_latent(self,latent):
		return self.discrimination(latent)
		


# define the NN architecture
class classifierTest(nn.Module):
	def __init__(self, n_class, hyp=None):
		super(classifierTest, self).__init__()
		self.n_class = n_class
		self._name = 'clfTest'

	@property
	def name(self):
		return self._name
	
	def build(self):
		f = 8
		self.CNN1 = nn.Conv2d(in_channels=1, kernel_size=(25,3),
			          out_channels=f, padding='same', bias=False)
		self.norm1 = nn.BatchNorm2d(f)
		self.maxP1 = nn.MaxPool2d((2,1))
		self.CNN2 = nn.Conv2d(in_channels=f, kernel_size=(5,3), out_channels=f,
			          padding='same', bias=True)
		self.flat1 = nn.Flatten()
		self.Lay1 = nn.Linear(1200, 50)
		self.Lay2 = nn.Linear(50, self.n_class)
	def forward(self, X):
		X = self.CNN1(X)
		X = torch.relu(X)
		X = self.maxP1(X)
		X = torch.relu(self.CNN2(X))
		X = self.flat1(X)
		X = torch.relu(self.Lay1(X))
		X= torch.softmax(self.Lay2(X),dim =1)
		return X

