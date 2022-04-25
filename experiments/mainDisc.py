import sys, argparse,os,glob

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerDisc import TLmodel

from Utils.myUtils import get_Clfparams, get_TLparams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='exp_name')
parser.add_argument('--trainClf', action='store_true')
parser.add_argument('--TLParamsFile', type=str, default=None)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--saveModel', type=bool, default=False)
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
	
	my_logger = WandbLogger(project='TL',
	                        log_model='all',
	                        name=args.expName + 'TL_' + args.source + '_to_' + args.target + 'DefExp')

if __name__ == '__main__':
	
	path_clf_params, path_TL_params = None, None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path,args.ClfParamsFile)
	if  args.TLParamsFile:
		path_TL_params = os.path.join(params_path, args.TLParamsFile)
	
	clfParams = get_Clfparams(path_clf_params)
	TLparams = get_TLparams(path_TL_params)

	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                input_shape=clfParams['input_shape'],
	                                batch_size=clfParams['bs'])
	dm_source.setup(split=False,normalize = True)
	file = f'mainModel_{args.source}'
	#if os.path.join(save_path,file + '_feature_extractor') not in glob.glob(save_path + '*'):
	if args.trainClf:
		trainer, clf, res = runClassifier(dm_source,clfParams)
		print('Source: ',res['train_acc'])
		clf.save_params(save_path,file)

	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                input_shape=clfParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=TLparams['bs'],
	                                type='target')
	dm_target.setup(split=False,normalize = True)
	if args.source =='Uschad':
		class_weight = torch.tensor([0.5,8,8,0.5])
	else:
		class_weight = None
	
	for i in range(args.trials):
		
		model = TLmodel(trainParams=TLparams,
						n_classes = args.n_classes,
		                lossParams = None,
		                save_path = None,
		                class_weight=class_weight,
		                model_hyp=clfParams)
	
		if my_logger:
			params = {}
			params['clfParams'] = clfParams
			params['SLparams'] = TLparams
			params['class_weight'] = class_weight
			my_logger.log_hyperparams(params)
			my_logger.watch(model)
	
		model.load_params(save_path,file)
		model.setDatasets(dm_source, dm_target)
		early_stopping = EarlyStopping('val_acc_target', mode='max', patience=7, verbose=True)
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=TLparams['epoch'],
		                  logger=my_logger,
		                  min_epochs = 1,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks = [early_stopping],
		                  multiple_trainloader_mode='max_size_cycle')
		
		trainer.fit(model)
		#model.save_params(save_path = )
		
		res = model.get_final_metrics()
		print(res)
		if my_logger:
			my_logger.log_metrics(res)
		if args.saveModel:
			trainer.save_checkpoint(f"../saved/FTmodel{args.source}_to_{args.target}_{args.expName}.ckpt")
		del model
		print('target: ', res['acc_target_all'],'  source: ', res['acc_source_all'])
		