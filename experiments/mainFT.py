import sys, argparse,os,glob

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
parser.add_argument('--expName', type=str, default='')
parser.add_argument('--trainClf', type=bool, default=True)
parser.add_argument('--TLParamsFile', type=str, default=None)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
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


def getHparams():
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['FE'] = 'fe2'
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = None
	
	clfParams['epoch'] = 14
	clfParams["dropout_rate"] = 0.2
	clfParams['bs'] = 128
	clfParams['lr'] = 0.00005
	clfParams['weight_decay'] = 0.1
	
	if args.ClfParamsFile:
		import json
		with open(os.path.join(params_path,args.TLParamsFile)) as f:
			aux = json.load(f)
		for k,v in aux.items():
			clfParams[k] = v

	if args.TLParamsFile:
		import json
		# with open(os.path.join(params_path,args.clfParamsFile)) as f:
		# 	clfParams = json.load(f)
		with open(os.path.join(params_path,args.TLParamsFile)) as f:
			TLparams = json.load(f)
		TLparams['gan'] = TLparams['gan'] =='True'
		return TLparams, clfParams
	

	TLparams = {}
	TLparams['lr'] =  0.001
	TLparams['gan'] = False
	TLparams['lr_gan'] = 0.0005
	TLparams['bs'] = 128
	TLparams['step_size'] = None
	TLparams['epoch'] = 50
	TLparams['feat_eng'] = 'sym'
	TLparams['alpha'] = 0.5
	TLparams['beta'] = 0.5
	TLparams['discrepancy'] = 'ot'
	TLparams['weight_decay'] = 0.1

	return TLparams, clfParams

if __name__ == '__main__':


	TLparams, clfParams = getHparams()
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                input_shape=clfParams['input_shape'],
	                                batch_size=clfParams['bs'])
	dm_source.setup(Loso=False, split=False,normalize = True)
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
	dm_target.setup(Loso=False, split=False,normalize = True)
	
	model = FTmodel(trainParams=TLparams,
					n_classes = args.n_classes,
	                lossParams = None,
	                save_path = None,
	                model_hyp=clfParams)

	if my_logger:
		params = {}
		params['clfParams'] = clfParams
		params['TLparams'] = TLparams
		my_logger.log_hyperparams(params)
		my_logger.watch(model)


	
	model.load_params(save_path,file)
	model.setDatasets(dm_source, dm_target)
	early_stopping = EarlyStopping('val_acc_target', mode='max', patience=10, verbose=True)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=TLparams['epoch'],
	                  logger=my_logger,
	                  progress_bar_refresh_rate=verbose,
	                  callbacks = [early_stopping],
	                  multiple_trainloader_mode='max_size_cycle')
	
	trainer.fit(model)
	res = model.get_final_metrics()
	print(res)
	if my_logger:
		my_logger.log_metrics(res)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/FTmodel{args.source}_to_{args.target}.ckpt")
