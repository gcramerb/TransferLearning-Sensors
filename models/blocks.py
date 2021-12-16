import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder1(nn.Module):
	"""

	"""
	def __init__(self,  hyp=None,
	             inputShape = (1,50,6)):
		super(Encoder1, self).__init__()
		self._name = 'Encoder'
		self.inputShape = inputShape
		if hyp:
			self.kernel_dim = hyp['kernel_dim']
			self.pooling_1 = (2, 3)
			self.pooling_2 = (5, 1)
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['encDim']
			# self.DropoutRate = hyp["DropoutRate"]


		self.n_win = 2
		self.CNN1 = nn.ModuleList([])
	## encoder layers ##
	def build(self):
		for i in range(self.n_win):
			self.CNN1.append(nn.Sequential(
				nn.Conv2d(in_channels=self.inputShape[0],
				          kernel_size=self.kernel_dim[i],
				          out_channels=self.n_filters[i],
				          padding='same', bias=True,
				          groups = self.inputShape[0]),
				
				nn.BatchNorm2d(self.n_filters[i]),
				nn.MaxPool2d(self.pooling_1),
				nn.LeakyReLU()
						)
					)
		
		self.CNN2 = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[0] + self.n_filters[1],
			          kernel_size=self.kernel_dim[0],
			          out_channels=self.n_filters[2],
			          padding='same', bias=True),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.MaxPool2d(self.pooling_2),
			nn.LeakyReLU()
		)
		
		self.DenseLayer = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[2],
			          kernel_size=self.kernel_dim[0],
			          out_channels=self.n_filters[3],
			          padding='same', bias=True),
			
			nn.BatchNorm2d(self.n_filters[3]),
			nn.LeakyReLU(),
			nn.Flatten(),
			nn.Linear(self.n_filters[3]*5*int(self.inputShape[-1]/3), self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.ReLU()
		)
	def forward(self,X):

		sensEncod = []
		for layer in self.CNN1:
			sensEncod.append(layer(X))

		sensors = torch.cat(sensEncod,1)
		sensors = self.CNN2(sensors)
		encoded = self.DenseLayer(sensors)
		return encoded


class Encoder2(nn.Module):

	def __init__(self, hyp=None,inputShape = (1,50,6)):
		super(Encoder2, self).__init__()
		self._name = 'Encoder'
		self.inputShape = inputShape
		if hyp:
			self.kernel_dim = hyp['kernel_dim']
			self.pooling_1 = (2, 3)
			self.pooling_2 = (5, 1)
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['encDim']


		self.n_win = 2
		self.CNN1 = nn.ModuleList([])
	
	## encoder layers ##
	def build(self):
		for i in range(self.n_win):
			self.CNN1.append(
				nn.Sequential(
				nn.Conv2d(in_channels=self.inputShape[0], kernel_size=self.kernel_dim[i],
				          out_channels=self.n_filters[i], padding='same', bias=True, groups=self.inputShape[0]),
				
				nn.BatchNorm2d(self.n_filters[i]),
				nn.MaxPool2d(self.pooling_1),
				nn.LeakyReLU()
					)
				)

		self.DenseLayer = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[0] + self.n_filters[1],
			          kernel_size=self.kernel_dim[0],
			          out_channels=self.n_filters[2],
			          padding='same', bias=True),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.LeakyReLU(),
			nn.MaxPool2d(self.pooling_2),
			nn.Flatten(),
			nn.Linear(self.n_filters[2]*5*int(self.inputShape[-1]/3), self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.LeakyReLU()
		)
	
	def forward(self, X):
		
		sensEncod = []
		for layer in self.CNN1:
			sensEncod.append(layer(X))
		sensors = torch.cat(sensEncod, 1)
		encoded = self.DenseLayer(sensors)
		return encoded


class Decoder(nn.Module):
	def __init__(self,
	             encoded_dim=64,
	             n_filters = None,
	             outputShape=(1, 50, 6)):
		super(Decoder, self).__init__()
		self._name = 'Decoder'
		self.encoded_dim = encoded_dim
		self.n_filters = n_filters
		self.convTrans_window = (5, 3)
		self.outputShape = outputShape
		if outputShape[-1] ==3:
			self.out_stride =2
		elif outputShape[-1] ==6:
			self.out_stride = 3
		self.pad_out = outputShape[0] - 1
		
	
	## decoder layers ##
	def build(self):
		self.linearDec = nn.Sequential(
			nn.Linear(self.encoded_dim, self.n_filters[2]*5*2),
			nn.LeakyReLU()
		)
		self.convDec = nn.Sequential(
			nn.ConvTranspose2d(in_channels=self.n_filters[2],
			                   kernel_size=self.convTrans_window,
			                   out_channels=self.n_filters[1],
			                   padding=(2, 1),
			                   output_padding=(1, 0),
			                   stride=(2, 1),
			                   dilation=(1, 1)
			                   ),
			# , ,output_padding=(1, 0)
			
			nn.BatchNorm2d(self.n_filters[1]),
			nn.LeakyReLU(),

			nn.ConvTranspose2d(in_channels=self.n_filters[1],
			                   kernel_size=self.convTrans_window,
			                   out_channels=self.outputShape[0],
			                   stride=(5, self.out_stride),
			                   padding=(0, self.pad_out),
			                   groups=self.outputShape[0]
			                   )
		)
	
	def forward(self, encoded):
		dec = self.linearDec(encoded)
		dec = dec.view(dec.shape[0], self.n_filters[2],5,2)
		dec = self.convDec(dec)
		return dec

class discriminator(nn.Module):
	def __init__(self,DropoutRate,encoded_dim,n_classes):
		super(discriminator, self).__init__()
		self.DropoutRate = DropoutRate
		self.encoded_dim = encoded_dim
		self.n_classes = n_classes

	def build(self):
		self.layer = nn.Sequential(
			nn.Dropout(p = self.DropoutRate ),
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
			nn.Linear(self.encoded_dim,1),
			nn.Sigmoid()
		)

	def forward(self, encoded):
		dec = self.linearDec(encoded)
		return dec
