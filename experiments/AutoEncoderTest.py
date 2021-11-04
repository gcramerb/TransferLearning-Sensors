import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse, pickle
import numpy as np

# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.autoencoder import ConvAutoencoder
from Utils.trainingConfig import EarlyStopping
from Utils.visualization import plotReconstruction as myPlot

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
showPlot = True
if args.slurm:
	n_ep = 350
	showPlot = False

else:
	n_ep = 150
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	from Utils.visualization import plotReconstruction as myPlot


def run():
	dm_source = CrossDatasetModule(data_dir=args.inPath, datasetName='Uschad', case='Source',
	                               batch_size=128)
	dm_source.setup(Loso=True)
	AE = ConvAutoencoder()
	AE.build()
	loss = torch.nn.MSELoss()
	
	epochs = n_ep
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	AE = AE.to(device).cuda()
	optimizer = optim.Adam(AE.parameters(), lr=0.01)
	scheduer = StepLR(optimizer, 30, gamma=0.5)
	
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
			
			rec_loss.mean().backward()
			optimizer.step()
			train_loss += rec_loss.mean().item()
		train_loss = train_loss / i
		scheduer.step()
		print(train_loss)
		Es(train_loss)
		if Es.early_stop:
			break
	
	with torch.no_grad():
		for i, batch in enumerate(dm_source.test_dataloader()):
			data, label = batch['data'], batch['label']
			sample = random.randint(0, len(data))
			AE = AE.to('cpu')
			enc, rec = AE(data)
			rec = rec[sample]
			rec[0] = rec[0] * dm_source.dataTest.std[0] + dm_source.dataTest.mean[0]
			rec[1] = rec[1] * dm_source.dataTest.std[1] + dm_source.dataTest.mean[1]
			true = data[sample]
			true[0] = true[0] * dm_source.dataTest.std[0] + dm_source.dataTest.mean[0]
			true[1] = true[1] * dm_source.dataTest.std[1] + dm_source.dataTest.mean[1]
			myPlot(rec, true, show=showPlot, file='rec_{args.source}.png')


if __name__ == '__main__':
	print(f"auto encoder for reconstruction of {args.source}")
	run()



