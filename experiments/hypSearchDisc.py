import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from mainDisc import runDisc
from trainers.trainerTL import TLmodel
import optuna
from Utils.myUtils import  MCI


import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	save_path = '../saved/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'

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
	clfParams['step_size'] = None
	clfParams['clf_epoch'] = None
	clfParams["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
	clfParams['bs'] = 128

	
	TLparams = {}
	TLparams['lr'] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
	TLparams['bs'] = 128
	TLparams['step_size'] = None
	TLparams['epoch']  = trial.suggest_int("epoch", 15, 120, step=15)
	TLparams['feat_eng'] = 'sym'
	TLparams['alpha'] = trial.suggest_float("alpha", 0.0, 1.0, step=0.1)
	TLparams['beta'] = trial.suggest_float("beta", 0.0005, 0.005, step=0.0005)
	TLparams['discrepancy'] = 'ot'
	TLparams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	
	return clfParams,TLparams
	


def objective(trial):
	# Initialize the best_val_loss value
	best_metric = float(-1)
	clfParams,TLparams = suggest_hyperparameters(trial)
	metrics = runDisc(clfParams,TLparams,args.source,args.target,1,save_path)
	acc =metrics['Target acc mean'][0]

	print('acc_target: ',acc)

	if acc >= best_metric:
		best_metric = acc
		print(f'Result: {args.source} to {args.target}')
		print('clfParams: ', clfParams,'\n')
		print('SLparams: ',TLparams,'\n\n')
	return best_metric


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print('Target dataset: ', args.target)
	print("  Trial number: ", study.best_trial.number)
	print("  Acc (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	run(200)
