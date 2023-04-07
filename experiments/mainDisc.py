import sys, argparse, os, glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerTL import TLmodel
from Utils.myUtils import MCI, getTeacherParams
from experiments.Utils.train import getDatasets,calculateMetricsFromTeacher

# seed = 2804
# print('Seeding with {}'.format(seed))
# torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='__')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--dicrepancy', type=str, default="")
parser.add_argument('--nClasses', type=int, default=4)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--savePath', type=str, default=None)
args = parser.parse_args()

my_logger = None
if args.slurm:
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'
else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
	#args.savePath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\teacherOficial\\'
	args.log = False
paramsPath = os.path.join(params_path, args.source[:3] + args.target[:3] + f"_{args.nClasses}activities.json")

if args.log:
	my_logger = WandbLogger(project='TransferLearning-Soft-Label',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)

if __name__ == '__main__':
	print(f"params loaded from: {paramsPath}")
	teacherParams = getTeacherParams(paramsPath)
	teacherParams['discrepancy'] = ""
	teacherParams['discrepancy'] = args.dicrepancy
	dm_source, dm_target = getDatasets(args.inPath,args.source,args.target,args.nClasses)
	model = runTeacher(teacherParams, dm_source, dm_target, args.trials, args.savePath, args.nClasses)
	metrics = calculateMetricsFromTeacher(model)
	print(metrics)
	if my_logger:
		my_logger.log_metrics(metrics)
