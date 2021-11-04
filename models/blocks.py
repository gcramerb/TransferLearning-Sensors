import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
	"""

	"""
	def __init__(self,  hyp=None):
		super(Encoder, self).__init__()
		self._name = 'Encoder'
		if hyp:
			self.conv_dim = hyp['conv_dim']
			self.pooling_1 = hyp['pooling_1']
			self.pooling_2 = hyp['pooling_2']
			self.n_filters = hyp['n_filters']
			self.encoded_dim = hyp['encDim']
			self.DropoutRate = hyp["DropoutRate"]

		else:
			self.conv_dim = [(5, 3), (25, 3)]
			self.pooling_1 = (2, 1)
			self.pooling_2 = (5, 3)
			self.n_filters = (8, 16, 32, 64)
			self.encoded_dim = 50
			self.DropoutRate = 0.0
		
		self.n_win = 2
		self.CNN1 = nn.ModuleList([])
	## encoder layers ##
	def build(self):
		for i in range(self.n_win):
			self.CNN1.append(nn.Sequential(
				nn.Conv2d(in_channels=2, kernel_size=self.conv_dim[i],
				          out_channels=self.n_filters[i], padding='same', bias=True,groups = 2),
				
				nn.BatchNorm2d(self.n_filters[i]),
				nn.LeakyReLU(),
				nn.MaxPool2d(self.pooling_1)))
		
		self.CNN2 = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[0] + self.n_filters[1], kernel_size=self.conv_dim[0],
			          out_channels=self.n_filters[2],
			          padding='same', bias=True),
			nn.BatchNorm2d(self.n_filters[2]),
			nn.LeakyReLU(),
			nn.MaxPool2d(self.pooling_2)
		)
		
		self.DenseLayer = nn.Sequential(
			nn.Conv2d(in_channels=self.n_filters[2], kernel_size=self.conv_dim[0], out_channels=self.n_filters[3],
			          padding='same', bias=True),
			nn.BatchNorm2d(self.n_filters[3]),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(self.n_filters[3]*5, self.encoded_dim),
			nn.BatchNorm1d(self.encoded_dim),
			nn.LeakyReLU()
			#nn.Dropout(p=self.DropoutRate, inplace=False)
		)
	def forward(self,X):

		sensEncod = []
		for layer in self.CNN1:
			sensEncod.append(layer(X))

		sensors = torch.cat(sensEncod,1)
		sensors = self.CNN2(sensors)
		encoded = self.DenseLayer(sensors)
		return encoded


class Decoder(nn.Module):
	def __init__(self, encoded_dim=50):
		super(Decoder, self).__init__()
		self._name = 'Decoder'
		self.encoded_dim = 50
		self.n_filters = (8, 16, 32, 64)
		self.convTrans_window = (5,3)

	## decoder layers ##
	def build(self):
		self.linearDec = nn.Sequential(
			nn.Linear(self.encoded_dim, self.n_filters[3]*5),
			nn.LeakyReLU()
		)
		self.convDec = nn.Sequential(
			#myReshape(()),
			nn.ConvTranspose2d(in_channels=self.n_filters[3], kernel_size=self.convTrans_window,
			                   out_channels=self.n_filters[2],padding=(0, 1),output_padding = (0,0),
			                    stride=(5, 1)),
			#, output_padding=(1, 0), dilation=(1, 1),
			
			nn.BatchNorm2d(self.n_filters[2]),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=self.n_filters[2], kernel_size=self.convTrans_window,
			                   out_channels=self.n_filters[1],output_padding = (1,0) ,padding=(4, 0),stride=(2, 1) ,groups = 2),
			#,  dilation=(2, 1)
			nn.BatchNorm2d(self.n_filters[1]),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=self.n_filters[1], kernel_size=self.convTrans_window, out_channels=2,
			                   stride=(1, 1),padding = (0,1)
			                   ),
			#,padding=(1, 0)
			#nn.LeakyReLU()
		)
	def forward(self,encoded):
		dec= self.linearDec(encoded)
		dec = dec.view(dec.shape[0],self.n_filters[3], 5, 1)
		dec = self.convDec(dec)
		return dec
