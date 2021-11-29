import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

from .blocks import Encoder1,Encoder2,Decoder,domainClf

# define the NN architecture
class ConvAutoencoder(nn.Module):
	"""
	
	"""
	def __init__(self, FeName = 'fe1',hyp=None,inputShape = (1,50,6)):
		super(ConvAutoencoder, self).__init__()
		self._name = 'AE'
		if FeName =='fe1':
			self.Encoder = Encoder1(hyp,inputShape=inputShape)
		else:
			self.Encoder = Encoder2(hyp,inputShape = inputShape)
		self.Decoder = Decoder(hyp['encDim'],
		                       n_filters = hyp['n_filters'],
		                       outputShape = inputShape)
		

	@property
	def name(self):
		return self._name
	def build(self):

		## encoder layers ##
		self.Encoder.build()
		self.Decoder.build()
		
		#from torchsummary import summary
		#summary(self.Decoder.to('cuda'), (1,80 ))




	def forward(self, X):
		encoded = self.Encoder(X)
		decoded = self.Decoder(encoded)
		return encoded,decoded
	

