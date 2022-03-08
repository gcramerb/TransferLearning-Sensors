import sys, argparse,os
import numpy as np
sys.path.insert(0, '../')
# import geomloss
from Utils.myUtils import get_Clfparams,get_foldsInfo,MCI

from pytorch_lightning.loggers import WandbLogger
from dataProcessing.dataModule import SingleDatasetModule
from trainers.runClf import runClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--source', type=str, default='Uschad')
args = parser.parse_args()

if args.slurm:
	verbose = 0
	inPath = '/storage/datasets/sensors/frankDatasets/'
	outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
	
	my_logger = WandbLogger(project='classifier',
	                        log_model='all',
	                        name=args.source + f'{args.n_classes}' + '_LOSO_clf')
else:
	verbose = 1
	inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	outPath = '../results/tests/'
	my_logger = None
	my_logger = WandbLogger(project='classifier',
	                        log_model='all',
	                        name=args.source + '_LOSO_clf')
	

if __name__ == '__main__':
	folds = get_foldsInfo()
	result = []
	result_train = []
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path, args.ClfParamsFile)
		clfParams = get_Clfparams(path_clf_params)
	else:
		clfParams = get_Clfparams()
	my_logger.log_hyperparams(clfParams)
	for fold_i in range(folds[args.source]):
	#fold_i = 1
		dm = SingleDatasetModule(data_dir=inPath,
		                         datasetName=args.source,
		                         n_classes=4,
		                         input_shape=clfParams['input_shape'],
		                         batch_size=clfParams['bs'])
		dm.setup(fold_i = fold_i,split=False, normalize=True)
		trainer, clf, res = runClassifier(dm, clfParams, my_logger=my_logger)
		result.append(res['test_acc'])
		result_train.append(res['train_acc'])

	print("test: ",result,'\n', MCI(result),'\n\n\n')
	print("train: ", result_train, '\n', MCI(result_train), '\n\n\n')
	my_logger.log_metrics(MCI(result_train))
	my_logger.log_metrics(MCI(result))


