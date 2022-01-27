import sys, argparse

sys.path.insert(0, '../')
# import geomloss
from Utils.Hparams import get_Clfparams

from pytorch_lightning.loggers import WandbLogger
from dataProcessing.dataModule import SingleDatasetModule
from train.runClf import runClassifier
if __name__ == '__main__':
	
	inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	source = 'Uschad'
	dm = SingleDatasetModule(data_dir=inPath,
	                         datasetName=source,
	                         n_classes=4,
	                         input_shape=(2, 50, 3),
	                         batch_size=128)
	dm.setup(split=False, normalize=True)
	trainer, clf, res = runClassifier(dm, get_Clfparams(), my_logger=None)
	print(res)
