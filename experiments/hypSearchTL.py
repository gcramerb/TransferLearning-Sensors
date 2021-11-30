import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
parser.add_argument('--expName', type=str, default='hypSrch')
parser.add_argument('--n_classes', type=int, default=4)

args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'


def suggest_hyperparameters(trial):
	params = {}
	params['lr_source'] =trial.suggest_float("lr_source", 1e-6, 1e-3, log=True)
	params['lr_target'] = trial.suggest_float("lr_target", 1e-4, 1e-2, log=True)
	params['bs_source'] = 128
	params['bs_target'] = 128
	params['step_size'] = 25
	params['n_epch'] = 1
	params['epch_rate'] = trial.suggest_int("epch_rate", 4, 16, step=4)
	params['alphaS'] = 0.5
	params['betaS'] = 0.5
	params['alphaT'] = 0
	params['discrepancy'] = 'ot'
	params['feat_eng'] = 'asym'
	params['weight_decay'] =trial.suggest_float("weight_decay", 0.0, 0.4, step=0.2)
	params['input_shape'] = (2, 50, 3)
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['encDim'] = trial.suggest_categorical("encDim", [48, 64])
	clfParams["DropoutRate"] = trial.suggest_float("DropoutRate", 0.0, 0.2, step=0.2)
	clfParams['FeName'] = 'fe2'
	
	lossParams = {}
	lossParams['blur'] = trial.suggest_float("blur", 0.005, 0.05, step=0.01)
	lossParams['scaling'] = trial.suggest_float("scaling", 0.5, 0.9, step=0.2)
	lossParams['debias'] = trial.suggest_categorical("debias", [True, False])
	return params, clfParams,lossParams



def objective(trial):
	# Initialize the best_val_loss value
	best_metric = float('Inf')
	
	trainParams,modelParams,lossParams = suggest_hyperparameters(trial)

	
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                inputShape=trainParams['input_shape'],
	                                batch_size=trainParams['bs_source'])
	dm_source.setup(Loso = False)
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                n_classes=args.n_classes,
	                                inputShape=trainParams['input_shape'],
	                                type = 'target',
	                                batch_size=trainParams['bs_target'])
	dm_target.setup(Loso = False,split = True)

	model = TLmodel(penalty=trainParams['discrepancy'],
	                alphaS=trainParams['alphaS'],
	                betaS = trainParams['betaS'],
	                alphaT = trainParams['alphaT'],
	                lr_source = trainParams['lr_source'],
	                lr_target=trainParams['lr_target'],
	                n_classes = args.n_classes,
	                data_shape = trainParams['input_shape'],
	                modelHyp = modelParams,
	                FeName = modelParams['FeName'],
	                weight_decay = trainParams['weight_decay'],
	                feat_eng = trainParams['feat_eng'],
	                epch_rate = trainParams['epch_rate'],
	                lossParams = lossParams)
	
	model.setDatasets(dm_source, dm_target)

	
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['n_epch'],
	                  #logger=my_logger,
	                  progress_bar_refresh_rate=0,
	                  #callbacks = [chkp_callback,early_stopping],
	                  multiple_trainloader_mode = 'max_size_cycle')
	
	
	trainer.fit(model)
	outcomes = trainer.validate(model=model)
	metric = outcomes[0]['valloss_AE']
	
	print(model.get_final_metrics())

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
	params = run(100)
