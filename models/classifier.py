import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

from .customLosses import MMDLoss, OTLoss
from .blocks import Encoder,discriminator

# define the NN architecture
class classifier(nn.Module):
	"""

	"""
	def __init__(self,
	             n_classes = 4,
	             trainParams=None,
	             input_shape = (2,50,3)):
		super(classifier, self).__init__()
		self.n_classes = n_classes
		self.trainParams = trainParams
		self.input_shape = input_shape
		

	def create_model(self):
		self.FE = Encoder(hyp=trainParams, input_shape=input_shape)
		self.FE.build()
		self.Disc = discriminator(dropout_rate = trainParams['dropout_rate'],
		                          encoded_dim = trainParams['enc_dim'],
		                          n_classes = self.n_classes)
		self.Disc.build()
	
	
	# from torchsummary import summary
		# summary(self.FE.to('cuda'), (2,50,3))

	def forward(self, X):
		encoded = self.FE.forward(X)
		pred = self.Disc(encoded)
		return encoded,pred
	def forward_from_latent(self,latent):
		return self.Disc(latent)
	def getLatent(self,X):
		return self.FE.forward(X)

