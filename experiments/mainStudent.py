import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, '../')
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from trainers.trainerTL import TLmodel
from Utils.myUtils import MCI, getTeacherParams, getStudentParams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='student')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--studentPath', action='store_true')
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--nClasses', type=int, default=4)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
teacherParamsPath = os.path.join(params_path, "Disc" + args.source[:3] + args.target[:3] + ".json")

if args.log:
	my_logger = WandbLogger(project='studentOficial',
	                        log_model='all',
	                        name=args.expName + f'{args.model}' + args.source + '_to_' + args.target)


if __name__ == '__main__':
	studentParams = getStudentParams()
	studentParams['input_shape'] = (2, args.freq * 2, 3)
	class_weight = None
	if args.nClasses ==4:
		class_weight = torch.tensor([0.5, 2, 2, 0.5])
	studentParams['class_weight'] = class_weight
	pre_train = True
	_, dm_target = getDatasets(args.inPath, "Dsads", args.target, args.nClasses)
	dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
	                                     datasetName=f"",
	                                     input_shape=2,
	                                     n_classes=args.nClasses,
	                                     batch_size=batchSize,
	                                     oneHotLabel=False,
	                                     shuffle=True)
	
	fileName = f"{args.source}_{args.target}pseudoLabel{args.model}.npz"
	dm_pseudoLabel.setup(normalize=False, fileName=fileName)
	metrics = runStudent(studentParams, dm_target, dm_pseudoLabel)
	print(metrics)
	if my_logger:
		my_logger.log_hyperparams(studentParams)
		my_logger.log_metrics(metrics)
