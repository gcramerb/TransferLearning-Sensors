import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy




sys.path.insert(0, "../Autoencoder/")
from modelUtils.custom_losses import SoftDTW
from modelUtils.custom_losses import My_dct_Loss as DCT_loss


# define the NN architecture
class ConvAutoencoder(nn.Module):
	"""
	
	"""
	def __init__(self, hyp=None):
		super(ConvAutoencoder, self).__init__()
		if hyp:
			conv_window = hyp['conv_window']
			pooling_window_1 = hyp['pooling_window_1']
			same_pad = pooling_window_1
			pooling_window_2 = hyp['pooling_window_2']
			n_filters = hyp['n_filters']
			encoded_dim = hyp['encDim']
		else:
			conv_window = (5, 3)
			pooling_window_1 = (2, 1)
			pooling_window_2 = (5, 1)
			n_filters = (8, 16, 32)
			encoded_dim = 50
		class myReshape(nn.Module):
			def __init__(self, newShape):
				super(myReshape, self).__init__()
			self.shape = newShape
			def forward(self, x):
				bs = x.shape[0]
				newShape =[bs] + self.shape
				return x.reshape(newShape)

		## encoder layers ##
		self.encoderSensor = nn.Sequential(
			nn.Conv2d(in_channels=1, kernel_size=conv_window, out_channels=n_filters[0], padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(pooling_window_1),
			nn.Conv2d(in_channels=n_filters[0], kernel_size=conv_window, out_channels=n_filters[1],padding='same'),
			nn.ReLU()
		)
		self.mergedSensors = nn.Sequential(
			nn.MaxPool2d(pooling_window_2),
			nn.Conv2d(in_channels=n_filters[1], kernel_size=conv_window, out_channels=n_filters[2], padding='same'),
			nn.ReLU()
			torch.flatten(),
			nn.linear(1000, encoded_dim),
			nn.ReLU()
		)
		
		## decoder layers ##
		self.decoder = nn.Sequential(
			nn.linear(encoded_dim, 1000),
			nn.ReLU(),
			myReshape(),
			nn.ConvTranspose2d(in_channels=1, kernel_size=conv_window, out_channels=n_filters[2],padding='valid',stride = (5,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=1, kernel_size=conv_window, out_channels=n_filters[2], padding='valid',stride=(2, 1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=1, kernel_size=conv_window, out_channels=n_filters[2], padding='valid',stride=(2, 1)),
			nn.ReLU()
		)

	def forward(self, X):
		encoded = []
		for sensor_data in X:
			encoded.append(self.encoderSensor(sensor_data))
		encoded = mergedSensors(torch.cat(tuple(encoded), 1))

		decoded = self.decoder(encoded)
		return decoded
