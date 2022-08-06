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
from Utils.myUtils import  MCI,getTeacherParams


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
	save_path = '../saved/hypDisc/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'

def suggest_hyperparameters(trial):
	Tparams = getTeacherParams()
	Tparams["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
	Tparams['lr'] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
	Tparams['epoch']  = trial.suggest_int("epoch", 15, 120, step=15)
	Tparams['alpha'] = trial.suggest_float("alpha", 0.0, 1.0, step=0.1)
	Tparams['beta'] = trial.suggest_float("beta", 0.005, 0.05, step=0.0005)
	Tparams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	return Tparams
	
finalResult = {}
finalResult['top 1'] = [0,{}]
finalResult['top 2'] = [0,{}]
finalResult['top 3'] = [0,{}]
finalResult['count'] = 0


def objective(trial):
	save = False
	teacherParams = suggest_hyperparameters(trial)
	
	if finalResult['count']%25 == 0:
		metrics = runDisc(teacherParams, args.source, args.target, 1, save_path, True)
		print(teacherParams)
		acc = metrics['Target acc mean'][0]
		print(f'Result: {args.source} to {args.target}: {acc}')
	else:
		metrics = runDisc(teacherParams, args.source, args.target, 1, save_path, False)
		acc = metrics['Target acc mean'][0]
	
	if acc>finalResult['top 1'][0]:
		finalResult['top 1'] = [acc,teacherParams]
	elif acc>finalResult['top 2'][0]:
		finalResult['top 2'] = [acc,teacherParams]
	elif acc>finalResult['top 3'][0]:
		finalResult['top 3'] = [acc,teacherParams]

		
	finalResult['count'] +=1
	return acc


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
		
	metrics = runDisc(finalResult['top 1'][1],args.source,args.target,1,save_path,True)
if __name__ == '__main__':
	run(401)
	print(finalResult)