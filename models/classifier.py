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
	             dropout_rate  = 0.0,
	             encoded_dim = 64,
	             input_shape =  None,
	             n_filters = None,
	             kernel_dim = None
					):
		super(classifier, self).__init__()
		self.input_shape = input_shape
		self.n_classes = n_classes
		self.encoded_dim = encoded_dim
		self.dropout_rate = dropout_rate
		self.trainParams = {}
		self.trainParams['kernel_dim'] = kernel_dim
		self.trainParams['n_filters'] = n_filters
		self.trainParams['enc_dim'] = encoded_dim

	def create_model(self):
		self.FE = Encoder(hyp=self.trainParams, input_shape=self.input_shape)
		self.FE.build()
		self.Disc = discriminator(dropout_rate = self.dropout_rate,
		                          encoded_dim = self.encoded_dim,
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

