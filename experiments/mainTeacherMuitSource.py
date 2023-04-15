import sys, argparse, os, glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule,MultiDatasetModule

import optuna
from trainers.trainerTL import TLmodel
from Utils.params import getTeacherParams
from Utils.train import getDatasets,calculateMetricsFromTeacher,runTeacher,suggestTeacherHyp

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--savePath', type=str, default=None)
args = parser.parse_args()
my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = f'/storage/datasets/sensors/frankDatasets_6actv/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	save_path = '../saved/hypDisc/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/V5/'

else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_6actv\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'

#
# if args.log:
# 	my_logger = WandbLogger(project='Disc',
# 	                        log_model='all',
# 	                        name=args.expName + args.source + '_to_' + args.target)
finalResult = {}
finalResult['top 1'] = [0, {}]
datasetList = ["Dsads","Ucihar","Uschad" ]
datasetList.remove(args.target)
if len(datasetList) != 2:
	raise ValueError('Dataset Name not exist')
	
dm_source = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                n_classes=6,
                                input_shape=2,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)

dm_source.setup(normalize=False,
                fileName=f"{datasetList[0]}_and_{datasetList[1]}.npz")
dm_target = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                input_shape=2,
                                n_classes=6,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)

dm_target.setup(normalize=False,
                fileName=f"{args.target}_MultiSource.npz")

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
def objective(trial):
	save = False
	teacherParams = suggestTeacherHyp(trial)
	teacherParams['discrepancy'] = "ot"
	model = runTeacher(teacherParams, dm_source, dm_target,6)
	predT = model.getPredict(domain='Target')
	predS = model.getPredict(domain='Source')
	metricsSource = calculateMetrics(predT['trueSource'], predT['predSource'])
	metricsTarget = calculateMetrics(predT['trueTarget'], predT['predTarget'])

	if metrics['Acc'] > finalResult['top 1'][0]:
		finalResult['top 1'] = [metricsTarget, teacherParams]
		print('\n------------------------------------------------\n')
		print(f'New Top 1: {metricsTarget["Acc"]} (Source: {metricsSource})\n')
		print(teacherParams)
		print('\n------------------------------------------------\n\n\n')
	return acc
if __name__ == '__main__':
	run(args.trials)
	print(finalResult)