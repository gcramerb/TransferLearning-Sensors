import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse, pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier, classifierTest
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss, OTLoss

# import geomloss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from collections import OrderedDict

from Utils.trainer_pl import networkLight
from Utils.trainerTL_pl import TLmodel
from dataProcessing.dataModule import CrossDatasetModule
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="mmd")
parser.add_argument('--batchS', type=int, default=128)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.nEpoch = 5
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def getHparams():
	clfParam = {}
	clfParam['kernel_dim'] = [(5, 3), (25, 1)]
	clfParam['n_filters'] = (4,8,16,24)
	clfParam['encDim'] =64
	clfParam["DropoutRate"] = 0.0
	return clfParam
	
	
if __name__ == '__main__':
	dm = CrossDatasetModule(data_dir=args.inPath)
	dm.setup(Loso = True)
	hparam = getHparams()
	model = TLmodel(penalty=args.penalty, alpha=args.alpha, lr=args.lr,modelHyp = hparam)
	trainer = Trainer(gpus=1, check_val_every_n_epoch=5,
	                  max_epochs=args.nEpoch, progress_bar_refresh_rate=1)
	trainer.fit(model, datamodule=dm)
	print(f"{args.source}_to_{args.target}\n")
	print(trainer.test(datamodule=dm))
