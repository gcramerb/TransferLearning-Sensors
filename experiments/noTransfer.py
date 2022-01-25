import sys, argparse

sys.path.insert(0, '../')

# import geomloss

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from train.trainerClf import networkLight
from dataProcessing.dataModule import SingleDatasetModule
from train.runClf import runClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

import numpy as np
import scipy.stats as st

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h
datasetList = ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def create_result_dict():
	result = {}
	for dat in datasetList:
		result[dat] = []
	return result


folds = {}
folds['Dsads'] = 8
folds['Uschad'] = 14
folds['Pamap2'] = 8
folds['Ucihar'] = 30


def getHparams():
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['FE'] = 'fe2'
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = None
	
	clfParams['epoch'] = 15
	clfParams["dropout_rate"] = 0.2
	clfParams['bs'] = 256
	clfParams['lr'] = 0.0001
	clfParams['weight_decay'] = 0.1
	return clfParams


if __name__ == '__main__':
	my_logger = WandbLogger(project='classifier',
	                        log_model='all',
	                        name=args.source + f'{args.n_classes}' + '_no_TL')
	result = create_result_dict()
	train_res = {}
	train_res[args.source] = []
	for fold_i in range(folds[args.source]):
		dm = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.source,
		                                n_classes=4,
		                                input_shape=(2, 50, 3),
		                                batch_size=128)
		dm.setup(fold_i=fold_i, split=False, normalize=True)
		trainer, clf, res = runClassifier(dm, getHparams(),my_logger = my_logger)


		result[args.source].append(res['test_acc'])
		train_res[args.source].append(res['train_acc'])
		for dataset in datasetList:
			if dataset != args.source:
				dm_target = SingleDatasetModule(data_dir=args.inPath,
				                         datasetName = dataset,
				                         n_classes=4,
				                         input_shape=(2, 50, 3),
				                         batch_size=128)
				dm_target.setup(split=False, normalize=True)
				res = clf.get_all_metrics(dm_target)
				result[dataset].append(res['val_acc'])
				del dm_target
		del trainer,dm,clf
	print('Resultado: ', result,'\n\n\n\n')
	for k,v in result.items():
		result[k] = mean_confidence_interval(v)
	print(result)
	train_res[args.source + '_train'] = mean_confidence_interval(train_res[args.source])
	my_logger.log_metrics(result)
	my_logger.log_metrics(train_res)
