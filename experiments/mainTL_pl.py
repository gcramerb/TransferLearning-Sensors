import sys, argparse

sys.path.insert(0, '../')

# import geomloss

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from train.trainerTL_pl import TLmodel
from dataProcessing.dataModule import SingleDatasetModule


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
	                        name=args.expName + '_' + args.source + '_to_' + args.target)

else:
	args.nEpoch = 50
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'
	my_logger = None
	

def getHparams(file_path = None):
	params = {}
	params['lr_source'] = 0.0005
	params['lr_target'] = 0.0002
	params['bs_source'] = 128
	params['bs_target'] = 128
	params['step_size'] = 25
	params['n_epch'] = 120
	params['epch_rate'] = 4
	params['alphaS'] = 0.5
	params['betaS'] = 0.5
	params['alphaT'] = 0
	params['discrepancy'] = 'ot'
	params['feat_eng'] = 'asym'
	params['weight_decay'] = 0.0
	params['input_shape'] = (2,50,3)
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4,16,18,24)
	clfParams['encDim'] = 64
	clfParams["DropoutRate"] = 0.2
	clfParams['FeName'] = 'fe2'

	if file_path:
		import json
		with open(file_path) as f:
			data = json.load(f)
		
		for k in data.keys():
			params[k] = data[k]
		clfParams['encDim'] = data['encDim']

	return params,clfParams

if __name__ == '__main__':
	trainParams, modelParams = getHparams(args.paramsPath)
	print(trainParams)

	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes = args.n_classes,
	                                inputShape=trainParams['input_shape'],
	                                batch_size=trainParams['bs_source'])
	dm_source.setup(Loso = False)
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                n_classes=args.n_classes,
	                                inputShape=trainParams['input_shape'],
	                                batch_size=trainParams['bs_target'])
	dm_target.setup(Loso = False,split = True)
	
	model = TLmodel(penalty=trainParams['discrepancy'],
	                alphaS=trainParams['alphaS'],
	                betaS = trainParams['betaS'],
	                alphaT = trainParams['alphaT'],
	                lr_source = trainParams['lr_source'],
	                lr_target=trainParams['lr_target'],
	                n_classes = args.n_classes,
	                data_shape = trainParams['input_shape'],
	                modelHyp = modelParams,
	                FeName = modelParams['FeName'],
	                weight_decay = trainParams['weight_decay'],
	                feat_eng = trainParams['feat_eng'],
	                epch_rate = trainParams['epch_rate'])
	if my_logger:
		my_logger.log_hyperparams(trainParams)
		my_logger.watch(model)
	#chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True )
	#early_stopping = EarlyStopping('trainloss_AE', mode='min', patience=10)
	model.setDatasets(dm_source, dm_target)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['n_epch'],
	                  logger=my_logger,
	                  progress_bar_refresh_rate=1,
	                  #callbacks = [chkp_callback,early_stopping],
	                  multiple_trainloader_mode = 'max_size_cycle')
	
	trainer.fit(model)
	#hat = model.predict(dm)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/TLmodel{args.source}_to_{args.target}_{trainParams['discrepancy']}.ckpt")
	print(f"{args.source}_to_{args.target}\n")
	#print(trainer.test(model = model,dataloaders=[dm.test_dataloader(),dm.train_dataloader()]))
	res = trainer.test(model=model)
	print(res)
	my_logger.log_metrics(res)
