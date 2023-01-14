import sys, argparse, os, glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerTL import TLmodel
from Utils.myUtils import MCI, getTeacherParams
from mainDisc import runDisc
import optuna
# seed = 2804
# print('Seeding with {}'.format(seed))
# torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='__')
parser.add_argument('--TLParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Uschad")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--freq', type=int, default=100)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--model', type=str, default="V5")
parser.add_argument('--savePath', type=str, default=None)
args = parser.parse_args()

paramsPath = None
finalResult = {}
finalResult['top 1'] = [0, {}]
if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/{args.model}/'
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\originalWindFreq\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\{args.model}\\'
	savedPath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\teacherOficialV5\\'


if args.TLParamsFile:
	paramsPath = os.path.join(params_path, args.TLParamsFile)
else:
	paramsPath = os.path.join(params_path, "Disc" + args.source[:3] + args.target[:3] + ".json")

dm_source = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                n_classes=args.n_classes,
                                freq=args.freq,
                                input_shape=2,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)
filename = f"{args.source}_{args.target}pseudoLabel{args.model}.npz"
dm_source.setup(normalize=False, fileName=filename)
dm_target = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                input_shape=2,
                                freq=args.freq,
                                n_classes=args.n_classes,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)
dm_target.setup(normalize=False, fileName=f"{args.target}AllOriginal_target_{args.source}AllOriginal.npz")


def suggest_hyperparameters(trial):
	clfParams = {}
	clfParams = getTeacherParams(paramsPath)
	clfParams['alpha'] = trial.suggest_float("alpha", 0.01, 2.0, step=0.05)
	clfParams['epoch'] = trial.suggest_int("epoch", 5, 81, step=25)
	clfParams["dropout_rate"] =  trial.suggest_float("dropout_rate", 0.0, 0.7, step=0.1)
	clfParams['bs'] = 128
	clfParams['lr'] =  trial.suggest_float("lr", 1e-5, 1e-3, log=True)
	clfParams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	return clfParams
def objective(trial):
	# Initialize the best_val_loss value
	clfParams= suggest_hyperparameters(trial)
	result,dictResult = runDisc(clfParams, dm_source, dm_target,1, None, False)
	print('Student acc Target: ',result)
	if  result >= finalResult['top 1'][0]:
		finalResult['top 1'] = [result, clfParams]
		print(f'Result: {args.source} to {args.target} --- {result}')
		print('clfParams: ', clfParams, '\n')
	return result
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
	print(f"params loaded from: {paramsPath}")
	run(1)
