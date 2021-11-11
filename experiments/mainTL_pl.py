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

from Utils.trainerClf_pl import networkLight
from Utils.trainerTL_pl import TLmodel
from dataProcessing.dataModule import CrossDatasetModule
import mlflow
#mlflow.set_tracking_uri("http://localhost:5000")

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Uschad")
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
	clfParam['kernel_dim'] = [(5, 3), (15, 1)]
	clfParam['n_filters'] = (2,4,8,16)
	clfParam['encDim'] =32
	clfParam["DropoutRate"] = 0.0
	clfParam['FeName'] = 'fe1'
	return clfParam

def getHparams():
	params = {}
	params['lr_source'] = 0.01
	params['lr_target'] = 0.001
	params['batch_size'] = 128
	params['n_epochs'] = 40
	params['alphaS'] = 0.2
	params['betaS'] = 0.05
	params['alphaT'] = 0.5
	params['discrepancy'] = 'ot'
	params['input_shape'] = (1,50,6)
	return params

if __name__ == '__main__':
	trainParams = getHparams()
	dm = CrossDatasetModule(data_dir=args.inPath,
	                        input_shape = trainParams['input_shape'],
	                        sourceName = args.source,
	                        targetName = args.target
							)
	dm.setup(Loso = False)
	hparam = getModelHparams()

	mlf_logger = MLFlowLogger(experiment_name='TL_trial1', save_dir='../results/mlflowTL/')
	#exp_id = mlf_logger.experiment.create_experiment('Exp 3')
	#mlf_logger.experiment.create_run(exp_id)
	mlf_logger.log_hyperparams(params=hparam)

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

	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=5,
	                  max_epochs=trainParams['n_epochs'],
	                  logger=mlf_logger,
	                  progress_bar_refresh_rate=1,
	                  callbacks = [chkp_callback])
	trainer.fit(model, datamodule=dm)
	#hat = model.predict(dm.test_dataloader())
	trainer.save_checkpoint(f"../saved/TLmodel{args.source}_to_{args.target}_{trainParams['discrepancy']}.ckpt")
	print(f"{args.source}_to_{args.target}\n")
	print(trainer.test(datamodule=dm))
