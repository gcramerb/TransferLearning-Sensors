import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim



import sys,os,time
import argparse
#from geomloss import SamplesLoss
sys.path.insert(0,'../')

from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss
from models.classifier import classifier
from dataProcessing.create_dataset import crossDataset,getData
from Utils.trainer import Trainer


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

def eval_and_log_metrics(prefix, actual, pred, epoch):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mlflow.log_metric("{}_rmse".format(prefix), rmse, step=epoch)
	return rmse

def suggest_hyperparameters(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 0.2, 0.8, step=0.1)
    bs = trial.suggest_int("bs",128,512,64)
    penalty = trial.suggest_categorical("loss", ["mmd", "ot"])
    return lr, alpha,bs,penalty


def objective(trial):
	# Initialize the best_val_loss value
	best_val_loss = float('Inf')
	
	# Start a new mlflow run
	with mlflow.start_run():
		# Get hyperparameter suggestions created by Optuna and log them as params using mlflow
		lr, alpha,bs,penalty = suggest_hyperparameters(trial)
		mlflow.log_params(trial.params)
		
		# Use CUDA if GPU is available and log device as param using mlflow
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		mlflow.log_param("device", device)
		
		# Initialize network
		hyp = {}
		hyp['model'] = 'clf'
		hyp['penalty'] = penalty
		hyp['lr'] = lr
		hyp['model_hyp'] = None
		
		net = Trainer(hyp)
		net.configTrain(bs=bs, alpha = alpha)
		if args.inPath is None:
			args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
		source = getData(args.inPath, args.source, True)
		target = getData(args.inPath, args.target, False)
		dataTrain = crossDataset(source, target)
		start = time.time()
		metric = net.train(dataTrain)
		end = time.time()
		timeToTrain = end - start
		if metric <= best_val_loss:
			best_val_loss = metric
		del source
		del target
		source = getData(args.inPath, args.source, getLabel=True)
		target = getData(args.inPath, args.target, getLabel=True)
		dataTest = crossDataset(source, target, targetLab=True)
		yTrueTarget, yTrueSource, yPredTarget, yPredSource = net.predict(dataTest)
		accTest = accuracy_score(yTrueTarget, yPredTarget)

		mlflow.log_metric("train_loss", metric, step=0)
		mlflow.log_metric("test_loss", accTest, step=0)
		mlflow.log_metric("timeTorTrain", timeToTrain, step=0)

	return best_val_loss

if __name__ == '__main__':
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize")
	study.optimize(objective, n_trials=3)
	
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

