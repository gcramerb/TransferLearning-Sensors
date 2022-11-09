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
	Tparams['enc_dim'] =  trial.suggest_categorical("enc_dim",[90,128])
	Tparams['lr'] = 0.0007
	Tparams['epoch']  = 75
	Tparams['input_shape'] = (2,100, 3)
	Tparams['alpha'] = trial.suggest_float("alpha", 0.01, 2.0, step=0.05)
	Tparams['beta'] = trial.suggest_float("beta", 0.005, 0.05, step=0.0005)
	Tparams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	f1 = trial.suggest_int("f1",4,12,step = 2)
	f2= trial.suggest_int("f2",12,24,step = 2)
	f3= trial.suggest_int("f3",24,32,step = 2)
	Tparams['n_filters'] = (f1, f2, f3)
	return Tparams
	
finalResult = {}
finalResult['top 1'] = [0,{}]
finalResult['top 2'] = [0,{}]
finalResult['top 3'] = [0,{}]
finalResult['count'] = 0
freq = 50
dm_source = SingleDatasetModule(data_dir=args.inPath,
                                datasetName=args.source,
                                n_classes=args.n_classes,
                                input_shape=(2, freq*2, 3),
                                batch_size=128,
                                freq=freq,
                                oneHotLabel=True,
                                shuffle=True)
dm_source.setup(normalize=True)
dm_target = SingleDatasetModule(data_dir=args.inPath,
                                datasetName=args.target,
                                input_shape=(2, freq*2, 3),
                                n_classes=args.n_classes,
                                batch_size=128,
								freq = freq,
							    oneHotLabel=True,
                                shuffle=True)
dm_target.setup(normalize=True)


def objective(trial):
	save = False
	teacherParams = suggest_hyperparameters(trial)
	metrics = runDisc(teacherParams, dm_source, dm_target, 1)
	acc = metrics['Target acc mean'][0]

	if acc>finalResult['top 1'][0]:
		finalResult['top 1'] = [acc,teacherParams]
		print('\n------------------------------------------------\n')
		print(f'New Top 1: {acc}\n')
		#metricsNew = runDisc(teacherParams, dm_source, dm_target, 1, save_path)
		#accNew = metricsNew['Target acc mean'][0]
		#print(f'Rerun Top 1 (saved): {args.source} to {args.target}: {accNew}\n')
		print(teacherParams)
		print('\n------------------------------------------------\n\n\n')
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
	return
if __name__ == '__main__':
	run(401)
	print(finalResult)
