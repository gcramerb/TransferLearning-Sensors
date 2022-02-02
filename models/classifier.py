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
	             n_classes,
	             hyp=None,
	             input_shape = (1,50,6)):
		super(classifier, self).__init__()
		self.n_classes = n_classes
		self.DropoutRate = hyp['dropout_rate']
		self.Encoder = Encoder(hyp=hyp, input_shape=input_shape)
		self.discrimination = discriminator(self.DropoutRate, hyp['enc_dim'], self.n_classes)
	
	@property
	def name(self):
		return self._name
	def build(self):

		self.Encoder.build()
		self.discrimination.build()
	
	# from torchsummary import summary
		# summary(self.Encoder.to('cuda'), (2,50,3))

	def forward(self, X):
		encoded = self.Encoder.forward(X)
		pred = self.discrimination(encoded)
		return encoded, pred
	def forward_from_latent(self,latent):
		return self.discrimination(latent)
		
