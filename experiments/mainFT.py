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
parser.add_argument('--expName', type=str, default='apr3')
parser.add_argument('--paramsPath', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	my_logger = WandbLogger(project='TL',
	                        log_model='all',
	                        name=args.expName + '_FT_' + args.source + '_to_' + args.target)
	save_path = '../results/saved/'
else:
	args.nEpoch = 50
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	my_logger = None
	args.paramsPath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\params1.json'
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'

def getHparams(file_path=None):
	params = {}
	params['lr_source'] = 0.00005
	params['lr_target'] = 0.001
	params['bs_source'] = 128
	params['bs_target'] = 128
	
	params['step_size'] = -1
	params['n_epch'] = 1

	params['alphaS'] = 0.2

	params['alphaT'] = None
	params['discrepancy'] = 'ot'

	params['weight_decay'] = 0.1
	params['input_shape'] = (2, 50, 3)
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['encDim'] = 64
	clfParams["DropoutRate"] = 0.2
	clfParams['FeName'] = 'fe2'
	
	if file_path:
		import json
		with open(file_path) as f:
			data = json.load(f)
		for k in data.keys():
			params[k] = data[k]
		if 'encDim' in data.keys():
			clfParams['encDim'] = data['encDim']
	
	return params, clfParams


if __name__ == '__main__':
	trainParams, modelParams = getHparams(args.paramsPath)

	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                inputShape=trainParams['input_shape'],
	                                batch_size=trainParams['bs_source'])
	dm_source.setup(Loso=False, split=False)

	trainer, clf, res = runClassifier(dm_source)
	clf.save_params(save_path)

	
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                n_classes=args.n_classes,
	                                type='target',
	                                inputShape=trainParams['input_shape'],
	                                batch_size=trainParams['bs_target'])
	dm_target.setup(Loso=False, split=False)
	
	model = FTmodel(penalty=trainParams['discrepancy'],
	                lr=trainParams['lr_target'],
	                n_classes=args.n_classes,
	                data_shape=trainParams['input_shape'],
	                modelHyp=modelParams,
	                FeName=modelParams['FeName'],
	                weight_decay=trainParams['weight_decay'])
	model.load_params(save_path)

	if my_logger:
		my_logger.log_hyperparams(trainParams)
		my_logger.log_hyperparams(modelParams)
		my_logger.watch(model)

	model.setDatasets(dm_source, dm_target)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['n_epch'],
	                  logger=my_logger,
	                  progress_bar_refresh_rate=1,
	                  # callbacks = [early_stopping],
	                  multiple_trainloader_mode='max_size_cycle')
	
	trainer.fit(model)
	res = trainer.test(model=model)
	print(res)
# my_logger.log_metrics(res)
