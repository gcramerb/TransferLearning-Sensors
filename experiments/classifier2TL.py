import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.utils.data import DataLoader, random_split

seed = 14
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

import sys, os, argparse, pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
np.random.seed(seed)
# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.classifier import classifier,classifierTest
from models.customLosses import MMDLoss
from dataProcessing.create_dataset import crossDataset, targetDataset, getData
from dataProcessing.dataModule import CrossDatasetModule

from Utils.myTrainer import myTrainer

import mlflow

# from pytorch_lightning.loggers import MLFlowLogger
# from pytorch_lightning import LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
#from Utils.trainerPL import  networkLight


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="ClDist")
parser.add_argument('--batchS', type=int, default=128)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--saveModel', type=bool, default=False)

args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.nEpoch = 5
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def getTrainSetup():
	trainSetup = dict()
	trainSetup['alpha'] = args.alpha
	trainSetup['nEpochs'] = args.nEpoch
	trainSetup['step_size'] = 30
	trainSetup['penalty'] = args.penalty
	trainSetup['lr'] = args.lr
	dm_source = CrossDatasetModule(data_dir=args.inPath, datasetName=args.source, case='Source',
	                               batch_size=args.batchS)
	dm_source.setup(Loso=True)
	dm_target = CrossDatasetModule(data_dir=args.inPath, datasetName=args.target, case='Target',
	                               batch_size=args.batchS)
	dm_target.setup(Loso=True)
	return trainSetup, dm_source, dm_target

def run():

	trainer = myTrainer('clf')

	trainSetup, dm_source, dm_target = getTrainSetup()
	trainer.setupTrain(trainSetup, dm_source, dm_target)
	trainHist = trainer.train()
	trainer.predict(metrics = True)

if __name__ == "__main__":
	
	run()
	
	for i in range(self.n_win):
		self.CNN1.append(nn.Sequential(
			nn.Conv2d(in_channels=1, kernel_size=self.conv_dim[i],
			          out_channels=self.n_filters[i], padding='same', bias=True),
			
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
		nn.Conv2d(in_channels=self.n_filters[3], kernel_size=self.conv_dim[0], out_channels=self.n_filters[3],
		          padding='same', bias=True),
		nn.BatchNorm2d(self.n_filters[3]),
		nn.ReLU(),
		# nn.Dropout(p=0.05, inplace=False),
		nn.Flatten(),
		nn.Linear(480, self.encoded_dim),
		nn.BatchNorm1d(self.encoded_dim),
		nn.LeakyReLU(),
		nn.Dropout(p=self.DropoutRate, inplace=False)
	)


def forward(self, X):
	AccEncoded = []
	GyrEncoded = []
	for layer in self.CNN1:
		AccEncoded.append(layer(X[:, :, :, 0:3]))
		GyrEncoded.append(layer(X[:, :, :, 3:6]))
	
	acc = torch.cat(AccEncoded, 1)
	gyr = torch.cat(GyrEncoded, 1)
	
	acc = self.CNN2(acc)
	gyr = self.CNN2(gyr)
	merged = torch.cat([acc, gyr], 1)
	encoded = self.DenseLayer(merged)
	return encoded
	
	
	
	
	
	
	
	
	
	
	
	