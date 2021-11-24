import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
parser.add_argument('--expName', type=str, default='trial_act_Def')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")

parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	verbose = 1
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
	clfParam['kernel_dim'] = [(5, 3), (25,3)]
	clfParam['n_filters'] = (4, 16, 18, 24)
	clfParam['encDim'] = 120
	clfParam["inputShape"] = (1, 50, 6)
	
	return clfParam


def getTrainHparms():
	trainParams = {}
	trainParams['nEpoch'] = 100
	trainParams['batch_size'] = 128
	trainParams['alpha'] = 0.5
	trainParams['lr'] = 0.0023318647476059827
	trainParams['step_size'] = 10
	return trainParams


if __name__ == '__main__':
	clfParams = getModelHparams()
	trainParams = getTrainHparms()
	
	dm = SingleDatasetModule(data_dir=args.inPath,
	                         datasetName=args.source,
	                         inputShape=clfParams["inputShape"],
	                         n_classes = args.n_classes,
	                         batch_size=trainParams['batch_size'])
	dm.setup(Loso = False)
	
	model = networkLight(alpha=trainParams['alpha'],
	                     lr=trainParams['lr'],
	                     inputShape = clfParams["inputShape"],
	                     FeName = 'fe2',
	                     step_size = trainParams['step_size'],
	                     modelHyp = clfParams)


	wandb_logger = WandbLogger(project='classifier', log_model='all',name = 'best_until_now')
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
	chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True)
	chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"

	trainer = Trainer(gpus=1,
	                  logger=wandb_logger,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['nEpoch'],
	                  progress_bar_refresh_rate=verbose,
	                  callbacks = [early_stopping])
	wandb_logger.watch(model)
	trainer.fit(model, datamodule=dm)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/clf1_{args.source}.ckpt")
	print(f"Training in {args.source} \n")
	print(trainer.test(model,datamodule = dm))