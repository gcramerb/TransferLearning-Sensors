import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, hyp=None,input_shape = (1,50,6)):
		super(Encoder, self).__init__()
		self._name = 'FE'
		self.input_shape = input_shape
		if hyp:
			self.kernel_dim = hyp['kernel_dim']
			self.pooling_1 = (2, 3)
			self.pooling_2 = (5, 1)
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['enc_dim']
		self.n_win = 2
		self.CNN1 = nn.ModuleList([])
	
	## encoder layers ##
	def build(self):
		
		fcl1 = self.n_filters[2] * int(self.input_shape[1]/10) * int(self.input_shape[-1] / 3)
		fcl2 = int(self.n_filters[2]/2) * int(self.input_shape[1]/10) * int(self.input_shape[-1] / 3)
		for i in range(self.n_win):
			self.CNN1.append(
				nn.Sequential(
				nn.Conv1d(in_channels=self.input_shape[0], kernel_size=self.kernel_dim[i],
				          out_channels=self.n_filters[i], padding='same', bias=True,
				          groups=self.input_shape[0]),
				
				nn.BatchNorm2d(self.n_filters[i]),
				nn.MaxPool2d(self.pooling_1),
				nn.LeakyReLU()
					)
				)
		self.DenseLayer = nn.Sequential(
			nn.Conv1d(in_channels=self.n_filters[0] + self.n_filters[1],
			          kernel_size=self.kernel_dim[0],
			          out_channels=self.n_filters[2],
			          padding='same', bias=True),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.LeakyReLU(),
			nn.MaxPool2d(self.pooling_2),
			nn.Flatten()
		)
	
	def forward(self, X):
		
		sensEncod = []
		for layer in self.CNN1:
			sensEncod.append(layer(X))
		sensors = torch.cat(sensEncod, 1)
		encoded = self.DenseLayer(sensors)
		return encoded

class discriminator(nn.Module):
	def __init__(self,dropout_rate,encoded_dim,n_classes):
		super(discriminator, self).__init__()
		self.dropout_rate = dropout_rate
		self.encoded_dim = encoded_dim
		self.n_classes = n_classes

	def build(self):
		self.layer = nn.Sequential(
			nn.Linear(90, self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.LeakyReLU(),
			nn.Dropout(p = self.dropout_rate ),
			nn.Linear(self.encoded_dim, self.n_classes),
			nn.Softmax(dim=1)
		)

	def forward(self,encoded):
		return self.layer(encoded)
	

class domainClf(nn.Module):
	def __init__(self,encoded_dim=64):
		super(domainClf, self).__init__()
		self._name = 'DomainCLf'
		self.encoded_dim = encoded_dim

	## decoder layers ##
	def build(self):
		self.linearDec = nn.Sequential(
			nn.Linear(self.encoded_dim,int(self.encoded_dim/2)),
			nn.LeakyReLU(),
			nn.Linear(int(self.encoded_dim/2), 1),
			nn.Sigmoid()
		)

	def forward(self, encoded):
		dec = self.linearDec(encoded)
		return dec
