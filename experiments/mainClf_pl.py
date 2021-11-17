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
from pytorch_lightning.loggers import MLFlowLogger,WandbLogger
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
parser.add_argument('--source', type=str, default="Ucihar")


parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

# def getModelHparams():
# 	clfParam = {}
# 	clfParam['kernel_dim'] = [(5, 3), (25, 1)]
# 	clfParam['n_filters'] = (6,8,24,24)
# 	clfParam['encDim'] =48
# 	clfParam["inputShape"] = (1,50,6)
#
# 	return clfParam
# def getTrainHparms():
# 	trainParams = {}
# 	trainParams['nEpoch'] = 30
# 	trainParams['batch_size'] = 64
# 	trainParams['alpha'] = 0.55
# 	trainParams['lr'] = 0.00015
# 	return trainParams
#
def getModelHparams():
	clfParam = {}
	clfParam['kernel_dim'] = [(5, 3), (15,3)]
	clfParam['n_filters'] = (4, 14, 24, 26)
	clfParam['encDim'] = 64
	clfParam["inputShape"] = (1, 50, 6)
	
	return clfParam


def getTrainHparms():
	trainParams = {}
	trainParams['nEpoch'] = 50
	trainParams['batch_size'] = 64
	trainParams['alpha'] = 0
	trainParams['lr'] = 0.001
	return trainParams


if __name__ == '__main__':
	clfParams = getModelHparams()
	trainParams = getTrainHparms()
	
	dm = SingleDatasetModule(data_dir=args.inPath,
	                         datasetName = args.source,
	                         batch_size = trainParams['batch_size'],
	                         inputShape = clfParams["inputShape"])
	dm.setup(Loso = False)
	
	model = networkLight(alpha=trainParams['alpha'],
	                     lr=trainParams['lr'],
	                     inputShape = clfParams["inputShape"],
	                     FeName = 'fe2',
	                     modelHyp = clfParams)
	early_stopping = EarlyStopping('val_loss', mode='min', patience=10)
	#mlf_logger = MLFlowLogger(experiment_name='nameLogger', save_dir='../results/mlflow/')
	wandb_logger = WandbLogger(project='classifier', log_model='all')
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
	chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True)
	chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"

	trainer = Trainer(gpus=1,
	                  logger=wandb_logger,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['nEpoch'],
	                  progress_bar_refresh_rate=1,
	                  callbacks = [early_stopping])
	wandb_logger.watch(model)
	trainer.fit(model, datamodule=dm)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/model1{args.source}.ckpt")
	print(f"Training in {args.source} \n")
	print(trainer.test(model,datamodule = dm))
	#mlf_logger.finalize()