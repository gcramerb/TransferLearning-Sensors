import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

from .blocks import Encoder,Decoder

# define the NN architecture
class ConvAutoencoder(nn.Module):
	"""
	
	"""
	def __init__(self, hyp=None):
		super(ConvAutoencoder, self).__init__()
		self._name = 'AE'
		self.Encoder = Encoder(hyp)
		self.Decoder = Decoder()


	
	@property
	def name(self):
		return self._name
	def build(self):

		## encoder layers ##
		self.Encoder.build()
		self.Decoder.build()
		# from torchsummary import summary
		# summary(self.Decoder.to('cuda'), (1,50 ))




	def forward(self, X):
		encoded = self.Encoder(X)
		decoded = self.Decoder(encoded)
		return encoded,decoded
	

