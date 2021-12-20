import sys, argparse

sys.path.insert(0, '../')

# import geomloss

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from train.runClf import runClassifier
from train.trainer_FT import FTmodel

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='clf_norm')
parser.add_argument('--input_shape', type=tuple, default=(2, 50, 3))
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--dataset', type=str, default="Pamap2")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	my_logger = WandbLogger(project='classifier',
	                        log_model='all',
	                        name=args.expName + args.dataset)
	save_path = '../results/saved/'
else:
	args.nEpoch = 50
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	my_logger = None
	args.paramsPath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\params1.json'
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'


if __name__ == '__main__':
	print('\n  =================== \n',args.dataset,' \n\n')
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.dataset,
	                                n_classes=args.n_classes,
	                                inputShape=args.input_shape,
	                                batch_size=args.bs)
	dm_source.setup(Loso=False, split=False, normalize=True)
	
	trainer, clf, res = runClassifier(dm_source,my_logger = my_logger)
	print(res)