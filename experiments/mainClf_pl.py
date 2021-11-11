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

from models.classifier import classifier,classifierTest
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss
#import geomloss

from pytorch_lightning import LightningDataModule, LightningModule,Trainer
from pytorch_lightning.callbacks import Callback,EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from collections import OrderedDict

from Utils.trainerClf_pl import networkLight
from dataProcessing.dataModule import SingleDatasetModule
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="ClDist")
parser.add_argument('--batchS', type=int, default=128)
parser.add_argument('--nEpoch', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.nEpoch = 50
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def getModelHparams():
	clfParam = {}
	clfParam['kernel_dim'] = [(5, 3), (15, 1)]
	clfParam['n_filters'] = (2,4,8,16)
	clfParam['encDim'] =32
	clfParam["DropoutRate"] = 0.0
	return clfParam

if __name__ == '__main__':
	dm = SingleDatasetModule(data_dir=args.inPath)
	dm.setup(Loso = False)
	
	model = networkLight(penalty=args.penalty,
	                     alpha=args.alpha,
	                     lr=args.lr,
	                     modelHyp = getModelHparams())
	mlf_logger = MLFlowLogger(experiment_name=args.expName, save_dir='../results/mlflow/')
	mlf_logger.log_hyperparams(params={'penalty': args.penalty, 'alpha': args.alpha,
	                                   'lr': args.lr, 'source': args.source})
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=10)
	chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True)
	chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"
	
	trainer = Trainer(gpus=1, logger=mlf_logger,
	                  check_val_every_n_epoch=5,
	                  max_epochs=args.nEpoch,
	                  progress_bar_refresh_rate=5)
	trainer.fit(model, datamodule=dm)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/model1{args.source}_to_{args.target}.ckpt")
	print(f"Training in {args.source} \n")
	print(trainer.test(model,datamodule = dm))
	mlf_logger.finalize()