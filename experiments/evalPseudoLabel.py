import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from pytorch_lightning.loggers import WandbLogger
sys.path.insert(0, '../')
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from Utils.myUtils import get_Stuparams

"""
The main idea of this experiment is to get the pseudo label of the target by the trained
models and evaluate it by the hold labels
"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='Stu_bench')
parser.add_argument('--TLParamsFile', type=str, default="DiscUscDsa.json")
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Uschad")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--PLmethod', type=str, default="simplest")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--saveModel', type=bool, default=True)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	save_path = '../saved/Disc/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
# my_logger = WandbLogger(project='TransferLearning-Soft-Label',
#                         log_model='all',
#                         name=args.expName + args.source + '_to_' + args.target)
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\Disc\\'


def analizePL(teacherParams,source, target):
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=teacherParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=teacherParams['bs'])
	
	dm_target.setup(normalize=True)
	teacher = ClfModel(lr=teacherParams['lr'],
	                   n_classes=args.n_classes,
	                   alpha=teacherParams['alpha'],
	                   step_size=teacherParams['step_size'],
	                   model_hyp=teacherParams,
	                   weight_decay=teacherParams['weight_decay'],
	                   class_weight=None,
	                   input_shape=teacherParams['input_shape'])
	
	file = f'Disc_class{source}_{target}'
	teacher.load_params(save_path, file)
	pred = teacher.predict(dm_target.test_dataloader())
	
	acc = accuracy_score(pred['true'], pred['pred'])
	print(f'INIT acc in Target{acc}')
	print(f"INIT number of samples: {len(pred['true'])}\n")
	
	for method in ['simplest','gmm','kernel']:
		print(f"METHOD: {method}\n")
		_,softLabel, trueLabel = getPseudoLabel(pred.copy(),method)
		print(f"number of samples: {len(trueLabel)}\n")
		acc = accuracy_score(trueLabel,softLabel)
		cm = confusion_matrix(trueLabel, softLabel)
		print(f'Acc: {acc}\n confusionMatrix: {cm}')
		del softLabel, trueLabel


	del teacher, dm_target
	return True
if __name__ == '__main__':
	path_TL_params = None
	if  args.TLParamsFile:
		path_TL_params = os.path.join(params_path, args.TLParamsFile)
	TLparams = get_Stuparams(path_TL_params)
	analizePL(TLparams,args.source, args.target)