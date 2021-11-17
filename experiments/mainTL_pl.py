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
from pytorch_lightning.loggers import MLFlowLogger,WandbLogger
from collections import OrderedDict

from Utils.trainerClf_pl import networkLight
from Utils.trainerTL_pl import TLmodel
from dataProcessing.dataModule import CrossDatasetModule,SingleDatasetModule
import mlflow
#mlflow.set_tracking_uri("http://localhost:5000")

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
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
	clfParam['kernel_dim'] = [(5, 3), (15, 3)]
	clfParam['n_filters'] = (4,14,24,26)
	clfParam['encDim'] =80
	clfParam["DropoutRate"] = 0.0
	clfParam['FeName'] = 'fe1'
	return clfParam
#Trial 39 finished with value: 0.9835648834705353 and parameters: {'lr': 2.2073152695750636e-05, 'alpha': 0.45000000000000007, 'bs_source': 32, 'step_size': 15, 'kernel2_1': 25, 'kernel2_2': 3, 'filter 1': 2, 'filter 2': 12, 'filter 3': 24, 'filter 4': 28, 'encDim': 96, 'FeName': 'fe2', 'dataFormat': (1, 50, 6)}.
def getHparams():
	params = {}
	params['lr_source'] = 0.003
	params['lr_target'] = 0.001
	params['bs_source'] = 64
	params['bs_target'] = 64
	#params['step_size'] = 25
	params['n_epochs'] = 100
	params['alphaS'] = 0.5
	params['betaS'] = 0.5
	params['alphaT'] = 3
	params['discrepancy'] = 'ot'
	params['input_shape'] = (2,50,3)
	return params

if __name__ == '__main__':
	trainParams = getHparams()
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                        inputShape = trainParams['input_shape'],
	                        datasetName = args.source,
	                        batch_size = trainParams['bs_source']
							)
	dm_source.setup(Loso = False)
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                        inputShape = trainParams['input_shape'],
	                        datasetName = args.target,
	                        batch_size = trainParams['bs_target']
							)
	dm_target.setup(Loso = False,split = True)
	
	hparam = getModelHparams()

	wandb_logger = WandbLogger(project='TL', log_model='all',name = 'first ')
	
	model = TLmodel(penalty=trainParams['discrepancy'],
	                alphaS=trainParams['alphaS'],
	                betaS = trainParams['betaS'],
	                alphaT = trainParams['alphaT'],
	                lr_source = trainParams['lr_source'],
	                lr_target=trainParams['lr_target'],
	                data_shape = trainParams['input_shape'],
	                modelHyp = hparam,
	                FeName = hparam['FeName'])
	chkp_callback = ModelCheckpoint(dirpath='../saved/',
	                                save_last=True )
	early_stopping = EarlyStopping('val_clf_loss', mode='min', patience=10)
	model.setDatasets(dm_source, dm_target)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=10,
	                  max_epochs=trainParams['n_epochs'],
	                  logger=wandb_logger,
	                  progress_bar_refresh_rate=1,
	                  callbacks = [chkp_callback,early_stopping],
	                  multiple_trainloader_mode = 'max_size_cycle')
	model.setDatasets(dm_source,dm_target)
	trainer.fit(model)
	#hat = model.predict(dm)
	
	#trainer.save_checkpoint(f"../saved/TLmodel{args.source}_to_{args.target}_{trainParams['discrepancy']}.ckpt")
	print(f"{args.source}_to_{args.target}\n")
	#print(trainer.test(model = model,dataloaders=[dm.test_dataloader(),dm.train_dataloader()]))
	print(trainer.test(model=model))
