import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
from sklearn.metrics import accuracy_score, recall_score, f1_score

import sys, os, time, pickle, argparse


sys.path.insert(0, '../')

from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss, OTLoss, classDistance
from models.classifier import classifier
from dataProcessing.dataModule import SingleDatasetModule

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from collections import OrderedDict

from Utils.trainerClf_pl import networkLight
from Utils.trainerTL_pl import TLmodel
from dataProcessing.dataModule import SingleDatasetModule
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--expName', type=str, default='trialDef')
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def getModelHparams():
	clfParam = {}
	clfParam['kernel_dim'] = [(5, 3), (15, 3)]
	clfParam['n_filters'] = (4,14,24,26)
	clfParam['encDim'] =80
	clfParam["DropoutRate"] = 0.0
	clfParam['FeName'] = 'fe2'
	return clfParam

def suggest_hyperparameters(trial):

	params = {}
	params['lr_source'] =trial.suggest_float("lr_source", 1e-4, 1e-1, log=True)
	params['lr_target'] = trial.suggest_float("lr_target", 1e-4, 1e-1, log=True)
	params['bs_source'] =  trial.suggest_categorical("bs_source", [64, 128])
	params['bs_target'] =  trial.suggest_categorical("bs_target", [64, 128])
	#params['step_size'] = 25
	params['n_epochs'] = 1
	params['alphaS'] = trial.suggest_float("alphaS", 0.0, 1.0, step=0.2)
	params['betaS'] = trial.suggest_float("betaS", 0.0, 1.0, step=0.2)
	params['alphaT'] = trial.suggest_float("alphaT", 0.0, 5.0, step=0.5)
	params['discrepancy'] = trial.suggest_categorical('discrepancy',['ot','mmd'])
	params['input_shape'] = (2,50,3)
	params['step_size'] = 10
	return params
	
	


def objective(trial):
	# Initialize the best_val_loss value
	best_metric = float('Inf')
	
	trainParams = suggest_hyperparameters(trial)
	modelHparams = getModelHparams()
	
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

	model = TLmodel(penalty=trainParams['discrepancy'],
	                alphaS=trainParams['alphaS'],
	                betaS = trainParams['betaS'],
	                alphaT = trainParams['alphaT'],
	                lr_source = trainParams['lr_source'],
	                lr_target=trainParams['lr_target'],
	                data_shape = trainParams['input_shape'],
	                modelHyp = modelHparams,
	                FeName = modelHparams['FeName'])
	
	model.setDatasets(dm_source, dm_target)
	early_stopping = EarlyStopping('val_clf_loss', mode='min', patience=10)
	
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=10,
	                  max_epochs=trainParams['n_epochs'],
	                  progress_bar_refresh_rate=1,
	                  callbacks = [early_stopping],
	                  multiple_trainloader_mode = 'max_size_cycle')
	
	
	trainer.fit(model)
	outcomes = trainer.validate(model=model)
	metric = outcomes[0]['val_AE_loss'] + outcomes[0]['val_clf_loss']
	if metric <= best_metric:
		best_metric = metric
	return best_metric


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Loss (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	params = run(1)
