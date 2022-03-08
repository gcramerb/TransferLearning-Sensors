import sys, argparse, os, glob

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule,CrossDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerSL import SLmodel

from Utils.myUtils import get_Clfparams, get_TLparams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='exp_name')
parser.add_argument('--trainClf', type=bool, default=False)
parser.add_argument('--TLParamsFile', type=str, default=None)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--saveModel', type=bool, default=True)
args = parser.parse_args()

my_logger = None
if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	save_path = '../saved/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
	my_logger = WandbLogger(project='TL',
	                        log_model='all',
	                        name=args.expName + '_FT_' + args.source + '_to_' + args.target)

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'

# my_logger = WandbLogger(project='TL',
#                         log_model='all',
#                         name=args.expName + 'TL_' + args.source + '_to_' + args.target)

if __name__ == '__main__':
	
	path_clf_params, path_TL_params = None, None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path, args.ClfParamsFile)
	if args.TLParamsFile:
		path_TL_params = os.path.join(params_path, args.TLParamsFile)
	
	if args.source == 'Uschad':
		class_weight = torch.tensor([0.5, 5, 5, 0.5])
	else:
		class_weight = None
	
	clfParams = get_Clfparams(path_clf_params)
	TLparams = get_TLparams(path_TL_params)
	
	iter = 5
	for i in range(iter):
		if i > 0:
			sl_path_file = os.path.join(args.inPath,f'{args.target}_pseudo_labels.npz')
		else:
			sl_path_file = None

		dm_source = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.source,
		                                n_classes=args.n_classes,
		                                input_shape=clfParams['input_shape'],
		                                batch_size=clfParams['bs'])
		dm_source.setup(split=False, normalize=True,SL_path_file =sl_path_file)
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.target,
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=TLparams['bs'],
		                                type='target')
		dm_target.setup(split=False, normalize=True)
		model = SLmodel(trainParams=TLparams,
		                trashold = 0.75,
		                n_classes=args.n_classes,
		                lossParams=None,
		                save_path=None,
		                class_weight=class_weight,
		                model_hyp=clfParams)
		model.setDatasets(dm_source, dm_target)
		model.create_model()

		if i%2 ==0:
			file = f'{args.source}_{args.target}_modelA'
		else:
			file = f'{args.source}_{args.target}_modelB'
		if i > 1:
			model.load_params(save_path, file)
	
		
		# early_stopping = EarlyStopping('val_acc_target', mode='max', patience=10, verbose=True)
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=5,
		                  min_epochs=1,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks=[],
		                  multiple_trainloader_mode='max_size_cycle')
		
		trainer.fit(model)
		model.generate_pseudoLab(path = args.inPath)
		model.save_params(save_path,file)
		del model,dm_source,dm_target,trainer
	
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                input_shape=clfParams['input_shape'],
	                                type = 'source',
	                                batch_size=clfParams['bs'])
	dm_source.setup(split=False, normalize=True)
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                input_shape=clfParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=TLparams['bs'],
	                                type='target')
	dm_target.setup(split=False, normalize=True)

	model = SLmodel(trainParams=TLparams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                save_path=None,
	                class_weight=None,
	                model_hyp=clfParams)
	model.setDatasets(dm_source, dm_target)
	model.create_model()
	if i % 2 == 0:
		file = f'{args.source}_{args.target}_modelA'
	else:
		file = f'{args.source}_{args.target}_modelB'
	model.load_params(save_path, file)
	outcomes = model.get_final_metrics()
	print("final Results: \n")
	print(outcomes)

