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
	setupTrain['alpha'] = trial.suggest_float("alpha", 0.05, 0.9, step=0.05)
	setupTrain['bs_source'] = trial.suggest_categorical("bs_source",[32,64,128,256,512])
	#setupTrain['bs_target'] = trial.suggest_categorical("bs_target", [32, 64, 128, 256, 512])
	setupTrain['bs_target'] = 512
	setupTrain['step_size'] =  trial.suggest_categorical("step_size",[40,50])
	
	#penalty = trial.suggest_categorical("loss", ["mmd", "ot"])
	setupTrain['penalty'] = 'ClDist'
	setupTrain['nEpochs'] = 300
	
	modelHyp = {}
	modelHyp['conv_dim'] =  [(5, 3), (25, 3)]
	modelHyp['pooling_1'] = (2, 1)
	modelHyp['pooling_2'] = (5, 1)
	modelHyp['n_filters'] = (8, 16, 32, 64)
	modelHyp['encDim'] = 50
	#modelHyp["DropoutRate"] = trial.suggest_float("DropoutRate", 0.0, 0.3, step=0.05)
	modelHyp["DropoutRate"] = 0.0
	return setupTrain,modelHyp


def objective(trial):
	# Initialize the best_val_loss value
	best_val_loss = float('Inf')
	setupTrain,hypModel = suggest_hyperparameters(trial)
	trainer = myTrainer('clf',hypModel)
	dm_source = CrossDatasetModule(data_dir=args.inPath, datasetName=args.source, case='Source',
	                               batch_size=setupTrain['bs_source'])
	dm_source.setup(Loso=True)
	dm_target = CrossDatasetModule(data_dir=args.inPath, datasetName=args.target, case='Target',
	                               batch_size=setupTrain['bs_target'])
	dm_target.setup(Loso=True)
	trainer.setupTrain(setupTrain, dm_source,dm_target)
	start = time.time()
	trainHist = trainer.train()
	outcomes = trainer.predict(stage = 'val',metrics=True)
	end = time.time()
	metric = outcomes['val_loss']
	timeToTrain = end - start
	if metric <= best_val_loss:
		best_val_loss = metric
	return best_val_loss
def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize")
	study.optimize(objective, n_trials=n_trials)
	# Start a new mlflow run
	# Print optuna study statistics
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ',args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Loss (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))
	mlflow.set_tracking_uri("../results/mlflow/")
	setupTrain, hypModel = suggest_hyperparameters(study.best_trial)
	for s in ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']:
		with mlflow.start_run(run_name=f'train_test in {s}'):
			# mlflow.log_params(study.best_trial.params)
			trainer = myTrainer('clf', hypModel)
			dm_source = CrossDatasetModule(data_dir=args.inPath, datasetName=s, case='Source',
			                               batch_size=setupTrain['bs_source'])
			dm_source.setup(Loso=True)
			dm_target = CrossDatasetModule(data_dir=args.inPath, datasetName=args.target, case='Target',
			                               batch_size=setupTrain['bs_target'])
			dm_target.setup(Loso=True)
			trainer.setupTrain(setupTrain, dm_source, dm_target)
			start = time.time()
			trainHist = trainer.train()
			stage = 'test'
			trainer.save(f'../saved/model_{s}.pkl')
			outcomes = trainer.predict(stage='test', metrics=True)
			end = time.time()
			mlflow.set_tag('Dataset_Source', s)
			with open("../results/mlflow/train_loss.pkl", "wb") as fp:
				pickle.dump(trainHist, fp)
			with open("../results/mlflow/val_loss.pkl", "wb") as fp:
				pickle.dump(trainer.valLoss, fp)
			mlflow.log_artifact(f"../results/mlflow/train_loss.pkl")
			mlflow.log_artifact(f"../results/mlflow/val_loss.pkl")

			mlflow.log_metric('acc_' + stage + '_Source', outcomes['acc_' + stage + '_Source'], step=0)
			mlflow.log_metric('acc_' + stage + '_Target', outcomes['acc_' + stage + '_Target'], step=0)

if __name__ == '__main__':
	params = run(200)
