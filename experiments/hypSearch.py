import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
from sklearn.metrics import accuracy_score, recall_score, f1_score



import sys,os,time
import argparse
#from geomloss import SamplesLoss
sys.path.insert(0,'../')

from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss,classDistance
from models.classifier import classifier
from dataProcessing.create_dataset import crossDataset, targetDataset, getData
from dataProcessing.dataModule import CrossDatasetModule
from Utils.myTrainer import myTrainer

from mlflow import pytorch
import mlflow
import optuna


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def eval_and_log_metrics(prefix, actual, pred, epoch):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mlflow.log_metric("{}_rmse".format(prefix), rmse, step=epoch)
	return rmse

def suggest_hyperparameters(trial):
	setupTrain = {}
	setupTrain['lr'] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
	setupTrain['alpha'] = trial.suggest_float("alpha", 0.1, 0.8, step=0.1)
	setupTrain['bs'] = trial.suggest_int("bs",64,128,256)
	setupTrain['step_size'] =  trial.suggest_int("bs",40,50)
	
	#penalty = trial.suggest_categorical("loss", ["mmd", "ot"])
	setupTrain['penalty'] = 'ClDist'
	setupTrain['nEpochs'] = 300
	
	return setupTrain


def objective(trial):
	# Initialize the best_val_loss value
	best_val_loss = float('Inf')
	
	# Start a new mlflow run
	with mlflow.start_run():
		# Get hyperparameter suggestions created by Optuna and log them as params using mlflow
		setupTrain = suggest_hyperparameters(trial)
		mlflow.log_params(trial.params)
		
		# Initialize network

		
		trainer = myTrainer('clf')

		dm = CrossDatasetModule(data_dir=args.inPath, source=args.source, target=args.target, batch_size=setupTrain['bs'])
		dm.setup()
		trainer.setupTrain(setupTrain, dm)
		start = time.time()
		trainHist = trainer.train()
		outcomes = trainer.predict(stage = 'val',metrics=True)
		end = time.time()
		metric = outcomes['val_loss']
		timeToTrain = end - start
		if metric <= best_val_loss:
			best_val_loss = metric

		mlflow.log_metric("val_loss", metric, step=0)
		mlflow.log_metric("last_train_loss", trainHist[-1], step=0)
		mlflow.log_metric("accTestSource",outcomes['accTestSource'] , step=0)
		mlflow.log_metric("accTestTarget", outcomes['accTestSource'], step=0)
		mlflow.log_metric("timeToTrain", timeToTrain, step=0)

	return best_val_loss
def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize")
	study.optimize(objective, n_trials=n_trials)
	
	# Print optuna study statistics
	print("\n++++++++++++++++++++++++++++++++++\n")
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	
	print("Best trial:")
	trial = study.best_trial
	
	print("  Trial number: ", trial.number)
	print("  Loss (trial value): ", trial.value)
	
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	run(1)


