import sys, argparse,os,glob
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerTL import TLmodel
from Utils.myUtils import get_Clfparams, get_TLparams, MCI

seed = 2809
print('Seeding with {}'.format(seed))
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='benchDisc_ntrials')
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
	save_path = '../saved/Disc/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\Disc\\'
if args.log:
	my_logger = WandbLogger(project='Disc',
	                        log_model='all',
	                        name= args.source + '_to_' + args.target + args.expName)


def runDisc(clfParams,TLparams,source,target,trials,save_path):
	final_result = {}
	final_result["Acc Target"] = []
	final_result["Acc Source"] = []
	if source == 'Uschad':
		class_weight = torch.tensor([0.5, 3, 3, 0.5])
	else:
		class_weight = None
	# if my_logger:
	# 	my_logger.log_hyperparams(clfParams)
	# 	my_logger.log_hyperparams(TLparams)
	# 	my_logger.log_hyperparams({'class_weight': class_weight})
	
	for i in range(trials):
		dm_source = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=source,
		                                n_classes=args.n_classes,
		                                input_shape=clfParams['input_shape'],
		                                batch_size=TLparams['bs'])
		dm_source.setup(normalize=True)
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=target,
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=TLparams['bs'])
		dm_target.setup(normalize=True)

		model = TLmodel(trainParams=TLparams,
		                n_classes=args.n_classes,
		                lossParams=None,
		                save_path=None,
		                class_weight=class_weight,
		                model_hyp=clfParams)
		
		# model.load_params(save_path,file)
		model.setDatasets(dm_source, dm_target)
		model.create_model()
		
		# from torchsummary import summary
		# summary(model.FE, (2, 50, 3))
		
		# early_stopping = EarlyStopping('val_loss', mode='min', patience=10, verbose=True)
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=TLparams['epoch'],
		                  logger=my_logger,
		                  min_epochs=1,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks=[],
		                  multiple_trainloader_mode='max_size_cycle')
		
		trainer.fit(model)
		predT = model.getPredict(domain='Target')
		predS = model.getPredict(domain='Source')
		accS = accuracy_score(predS['trueSource'], predS['predSource'])
		accT = accuracy_score(predT['trueTarget'], predT['predTarget'])
		print('Source: ', accS, '  Target: ', accT)
		final_result["Acc Target"].append(accT)
		final_result["Acc Source"].append(accS)
		model.save_params(save_path, f'Disc_{source}_{target}')
		del model, trainer, dm_target, dm_source
	final_result['Target acc mean'] = MCI(final_result["Acc Target"])
	final_result['Source acc mean'] = MCI(final_result["Acc Source"])
	return final_result

if __name__ == '__main__':
	path_clf_params, path_TL_params = None, None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path,args.ClfParamsFile)
	if  args.TLParamsFile:
		path_TL_params = os.path.join(params_path, args.TLParamsFile)
	
	clfParams = get_Clfparams(path_clf_params)
	TLparams = get_TLparams(path_TL_params)
	
	final_result = runDisc(clfParams,TLparams,args.source,args.target,args.trials,save_path)

	print(final_result)
	if my_logger:
		my_logger.log_metrics(final_result)
