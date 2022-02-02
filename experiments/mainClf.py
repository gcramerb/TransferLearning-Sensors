import sys, argparse
import numpy as np
sys.path.insert(0, '../')
# import geomloss
from Utils.myUtils import get_Clfparams,get_foldsInfo,MCI

from pytorch_lightning.loggers import WandbLogger
from dataProcessing.dataModule import SingleDatasetModule
from trainers.runClf import runClassifier

if __name__ == '__main__':
	source = 'Uschad'
	folds = get_foldsInfo()
	inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	result = []
	result_train = []
	for fold_i in range(folds[source]):
		dm = SingleDatasetModule(data_dir=inPath,
		                         datasetName=source,
		                         n_classes=4,
		                         input_shape=(2, 50, 3),
		                         batch_size=128)
		dm.setup(fold_i = fold_i,split=False, normalize=True)
		trainer, clf, res = runClassifier(dm, get_Clfparams(), my_logger=None)
		result.append(res['test_acc'])
		result_train.append(res['train_acc'])
		
	print(result,'\n\n\n\n')
	print(result_train, '\n\n\n\n')
	print(MCI(result))
	print(MCI(result_train))
