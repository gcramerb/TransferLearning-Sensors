import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim



import sys, os,argparse,pickle
import numpy as np

# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.autoencoder import ConvAutoencoder
from Utils.trainingConfig import EarlyStopping


from dataProcessing.dataModule import CrossDatasetModule

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="mmd")
args = parser.parse_args()

if args.slurm:
	n_ep = 300
else:
	n_ep = 50
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'

def run():
	dm_source = CrossDatasetModule(data_dir=args.inPath, datasetName='Uschad', case='Source',
	                               batch_size=128)
	dm_source.setup(Loso=True)
	AE = ConvAutoencoder()
	AE.build()
	loss = torch.nn.MSELoss()
	
	epochs =n_ep
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	AE= AE.to(device).cuda()
	optimizer = optim.Adam(AE.parameters(), lr=0.01)
	scheduer = StepLR(optimizer,50, gamma=0.5)
	
	Es = EarlyStopping(patience=10)
	
	for epoch in range(epochs):
		# monitor training loss
		train_loss = 0.0
		for i, batch in enumerate(dm_source.train_dataloader()):
			optimizer.zero_grad()
			data, label = batch['data'], batch['label']
			# we can put the data in GPU to process but with 'no_grad' pytorch way?
			data = data.to(device, dtype=torch.float)
			encoded, decoded = AE.forward(data)
			rec_loss = loss(data, decoded)

			rec_loss.backward()
			optimizer.step()
			train_loss += rec_loss.mean().item()
		train_loss = train_loss / i
		print(train_loss)
		Es(train_loss)
		if Es.early_stop:
			break
if __name__ == '__main__':
	print(f"auto encoder for reconstruction of {args.source}")
	run()



