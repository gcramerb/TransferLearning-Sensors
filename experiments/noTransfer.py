import sys, argparse

sys.path.insert(0, '../')

# import geomloss

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from train.trainerClf_pl import networkLight
from dataProcessing.dataModule import SingleDatasetModule
from train.runClf import runClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()
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

if __name__ == '__main__':
	
	
	my_logger = WandbLogger(project='classifier',
	                        log_model='all',
	                        name=args.source + f'{args.n_classes}'+ '_no_TL')
	
	result = create_result_dict()
	dm = SingleDatasetModule(data_dir=args.inPath, datasetName=args.source, n_classes=args.n_classes,
	                         input_shape=(2, 50, 3), batch_size=args.batch_size)
	dm.setup(Loso=True)
	folds_ = dm.get_n_folds()
	folds_ = [0]
	for fold_i in folds_:
		dm.set_fold(fold_i)
		trainer, modelTrainded,metrics = runClassifier(dm)
		print(f"Training in {args.source} \n")
		result[args.source].append(metrics[0]['val_acc'])
		print('train acc: ')
		print(model.get_train_metics(dm.train_dataloader()))
		for dataset in datasetList:
			if dataset != args.source:
				
				dm_target = SingleDatasetModule(data_dir=args.inPath, datasetName=dataset, n_classes=args.n_classes,
				                                input_shape=(2, 50, 3), batch_size=args.batch_size, type='target')
				
				dm_target.setup(split=False)
				res = trainer.validate(model, datamodule=dm_target)
				result[dataset].append(res[0]['val_acc'])
				del dm_target
		del trainer
		del model
	for k,v in result.items():
		result[k] = mean_confidence_interval(v)
	print(result)
	my_logger.log_metrics(result)
