import sys, argparse, os, glob
import numpy as np

sys.path.insert(0, '../')

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from mainTS import runTS
from trainers.trainerTL import TLmodel
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	save_path = '../saved/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'


def suggest_hyperparameters(trial):
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = None
	clfParams['clf_epoch'] = trial.suggest_int("epoch", 3, 18, step=3)
	clfParams["dropout_rate"] =  trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
	clfParams['bs'] = 128
	clfParams['clf_lr'] =  trial.suggest_float("clf_lr", 1e-5, 1e-3, log=True)
	clfParams['weight_decay'] = trial.suggest_float("Clfweight_decay", 0.0, 0.9, step=0.1)
	
	SLparams = {}
	SLparams['bs'] = 128
	SLparams['step_size'] = None
	SLparams['iter'] =  10
	SLparams['trasholdStu'] = trial.suggest_float("trasholdStu", 0.5, 0.95, step=0.05)
	return clfParams, SLparams


def objective(trial):
	# Initialize the best_val_loss value
	best_metric = float(-1)
	clfParams, SLparams = suggest_hyperparameters(trial)
	metrics = runTS(clfParams, SLparams, 'hypParam',,
	result = np.max(metrics['Student acc in Target'])
	print('Student acc Target: ',result)
	if result >= best_metric:
		best_metric = result
		print(f'Result: {args.source} to {args.target}')
		print('clfParams: ', clfParams, '\n')
		print('SLparams: ', SLparams, '\n\n\n')
	return best_metric


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print('Target dataset: ', args.target)
	print(" Trial number: ", study.best_trial.number)
	print("  Acc (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	run(300)
