import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
from sklearn.metrics import accuracy_score, recall_score, f1_score

import sys,os,time, pickle, argparse
#from geomloss import SamplesLoss
sys.path.insert(0,'../')

from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss,classDistance
from models.classifier import classifier
from dataProcessing.dataModule import SingleDatasetModule

from pytorch_lightning import LightningDataModule, LightningModule,Trainer
from pytorch_lightning.callbacks import Callback,EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from collections import OrderedDict

from Utils.trainerClf_pl import networkLight
from dataProcessing.dataModule import SingleDatasetModule
import mlflow
import optuna


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'


def suggest_hyperparameters(trial):
	setupTrain = {}
	setupTrain['lr'] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
	setupTrain['alpha'] = trial.suggest_float("alpha", 0.05, 2.5, step=0.05)
	setupTrain['bs'] = trial.suggest_categorical("bs_source",[32,64,128,256,512])
	setupTrain['step_size'] =  trial.suggest_categorical("step_size",[20,40,50])
	setupTrain['nEpochs'] = 1
	
	
	modelHyp = {}
	kernel =  trial.suggest_int('kernel 2_1',15,25,step=5)
	modelHyp['kernel_dim'] =  [(5, 3), (kernel, 1)]
	f1 = trial.suggest_int('filter 1',4,8,step=2)
	f2 = trial.suggest_int('filter 2',8,16,step=2)
	f3 = trial.suggest_int('filter 3',16,24,step=2)
	f4 = trial.suggest_int('filter 4',32,48,step=2)
	modelHyp['n_filters'] = (f1,f2, f3, f4)
	modelHyp['encDim'] =  trial.suggest_int('encDim',30,120,step=10)
	modelHyp["DropoutRate"] = 0.0
	FeName = trial.suggest_categorical('FeName', ['fe1', 'fe2'])
	inputShape = trial.suggest_categorical('dataFormat', [(1,50,6), (2,50,3)])
	
	return setupTrain, modelHyp, FeName, inputShape


def objective(trial):
	# Initialize the best_val_loss value
	best_val_loss = float('Inf')
	setupTrain, hypModel, FeName, inputShape = suggest_hyperparameters(trial)

	dm = SingleDatasetModule(data_dir=args.inPath,
	                         datasetName=args.source,
	                         batch_size=setupTrain['bs'],
	                         inputShape = inputShape)
	dm.setup(Loso=False)
	
	
	model = networkLight(alpha=setupTrain['alpha'],
	                     lr=setupTrain['lr'],
	                     FeName = FeName,
	                     modelHyp = hypModel,
	                     inputShape = inputShape)
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=10)
	
	mlf_logger = MLFlowLogger(experiment_name='hypSearch', save_dir='../results/mlflow/')
	trainer = Trainer(gpus=1, logger=mlf_logger,
	                  check_val_every_n_epoch=5,
	                  max_epochs=setupTrain['nEpochs'],
	                  progress_bar_refresh_rate=5,
	                  callbacks = [early_stopping])
	trainer.fit(model, datamodule=dm)
	outcomes = trainer.validate(model = model,dataloaders=dm.val_dataloader())
	metric = outcomes[0]['val_loss']
	if metric <= best_val_loss:
		best_val_loss = metric
	return best_val_loss

def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize")
	study.optimize(objective, n_trials=n_trials)
	
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ',args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Loss (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))
	mlflow.set_tracking_uri("../results/mlflow/")
	setupTrain, hypModel, FeName, inputShape = suggest_hyperparameters(study.best_trial)
	
	
	for s in ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']:
		with mlflow.start_run(run_name=f'train_test in {s}'):
			mlflow.log_params(study.best_trial.params)
			dm_source = SingleDatasetModule(data_dir=args.inPath,
			                                datasetName=s,
			                                batch_size=setupTrain['bs'],
			                                inputShape = inputShape)
			dm_source.setup(Loso=False)
			model = networkLight(alpha=setupTrain['alpha'],
			                     lr=setupTrain['lr'],
			                     modelHyp=hypModel,
			                     FeName = FeName,
			                     inputShape=inputShape)
			
			early_stopping = EarlyStopping('val_loss', mode='min', patience=10)
			trainer = Trainer(gpus=1,
			                  check_val_every_n_epoch=5,
			                  max_epochs=setupTrain['nEpochs'],
			                  progress_bar_refresh_rate=5,
			                  callbacks=[early_stopping])
			trainer.fit(model, datamodule=dm)
			outcomes = trainer.test(model,dataloaders=dm.test_dataloader())
			outcomes = outcomes[0]
			print(outcomes)

			mlflow.set_tag('Dataset_Source', s)
			for k,v in outcomes.items():
				mlflow.log_metric(k,v, step=0)

if __name__ == '__main__':
	params = run(2)
