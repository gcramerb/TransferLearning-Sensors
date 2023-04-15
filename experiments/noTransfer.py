import sys, argparse,os
import numpy as np
import scipy.stats as st

sys.path.insert(0, '../')
from Utils.params import get_Clfparams,get_TLparams,get_foldsInfo,MCI
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from dataProcessing.dataModule import SingleDatasetModule
from trainers.runClf import runClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

datasetList = ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']
my_logger = WandbLogger(project='classifier',
                        log_model='all',
                        name=args.source + f'{args.n_classes}' + 'just_clf')
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def create_result_dict():
	result = {}
	for dat in datasetList:
		result[dat] = []
	return result

if __name__ == '__main__':
	"""
	This experiment assumes that you alread have a good classifier for your source data.
	
	"""
	folds = get_foldsInfo()
	if args.ClfParamsFile:
		clfParams = get_Clfparams(os.path.join(params_path,args.ClfParamsFile))
	else:
		clfParams = get_Clfparams()

	my_logger.log_hyperparams(clfParams)
	result = create_result_dict()
	for i in range(folds[args.source]):
		dm = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.source,
		                                n_classes=args.n_classes,
		                                input_shape=clfParams['input_shape'],
		                                batch_size=clfParams['bs'])
		
		dm.setup(fold_i = i, normalize=True)
	#for i in range(args.trials):
		trainer, clf, res = runClassifier(dm,clfParams ,my_logger = my_logger)
		result[args.source].append(res['test_acc'])
		
		# for dataset in datasetList:
		# 	if dataset != args.source:
		# 		dm_target = SingleDatasetModule(data_dir=args.inPath,
		# 		                         datasetName = dataset,
		# 		                         n_classes=4,
		# 		                         input_shape=clfParams['input_shape'],
		#                                 batch_size=clfParams['bs'])
		# 		dm_target.setup(split=False, normalize=True)
		# 		res = clf.get_all_metrics(dm_target)
		# 		result[dataset].append(res['test_acc'])
		# 		del dm_target
		del trainer,clf,dm
	print('Resultado: ', result,'\n\n')
	# for k,v in result.items():
	# 	result[k] = MCI(v)
	result[args.source] = MCI(result[args.source])
	print('Resultado: ', result, '\n\n')
	my_logger.log_metrics(result)