import sys, argparse, os, glob
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule,CrossDatasetModule
sys.path.insert(0, '../')
from trainers.runClf import runClassifier
from trainers.trainerSL import SLmodel
from Utils.myUtils import get_Clfparams, get_TLparams,get_SLparams
from models.pseudoLabSelection import saveSL

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='mainSL')
parser.add_argument('--trainClf', type=bool, default=False)
parser.add_argument('--SLParamsFile', type=str, default=None)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Pamap2")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--saveModel', type=bool, default=True)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	save_path = '../saved/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
	my_logger = WandbLogger(project='TransferLearning-Soft-Label',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'



if __name__ == '__main__':
	
	path_clf_params, path_SL_params = None, None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path, args.ClfParamsFile)
	if args.SLParamsFile:
		path_SL_params = os.path.join(params_path, args.SLParamsFile)
	
	if args.source == 'Uschad':
		class_weight = torch.tensor([0.5, 5, 5, 0.5])
	else:
		class_weight = None
	
	clfParams = get_Clfparams(path_clf_params)
	SLparams = get_SLparams(path_SL_params)
	first_save = True
	sl_path_file = None
	source_metric_i = []
	target_metric_i = []
	num_samples = []
	for i in range(SLparams['iter']):

		dm_source = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.source,
		                                n_classes=args.n_classes,
		                                input_shape=clfParams['input_shape'],
		                                batch_size=SLparams['bs'])
		dm_source.setup(normalize=True,SL_path_file =sl_path_file)
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.target,
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=SLparams['bs'])
		dm_target.setup(normalize=True)
		model = SLmodel(trainParams=SLparams,
		                trashold = SLparams['trasholdDisc'],
		                n_classes=args.n_classes,
		                lossParams=None,
		                save_path=None,
		                class_weight=class_weight,
		                model_hyp=clfParams)
		model.setDatasets(dm_source, dm_target)
		model.create_model()
		if my_logger:
			my_logger.watch(model)

		file = f'{args.source}_{args.target}_model{i%2}'
		if i > 0:
			model.load_params(save_path, file)
			sl_path_file = os.path.join(args.inPath, f'{args.target}_to_{args.n_classes}_mainSL.npz')
			first_save = False

		# early_stopping = EarlyStopping('val_acc_target', mode='max', patience=10, verbose=True)
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=SLparams['epoch'],
		                  min_epochs=1,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks=[],
		                  multiple_trainloader_mode='max_size_cycle')
		
		trainer.fit(model)
		
		pred = model.getPredict(domain='Target')
		ns = saveSL(path_file=sl_path_file, data = pred['data'],
		            probs = pred['probs'],trh = SLparams['trasholdDisc'],
		            first_save = first_save)
		target_metric_i.append(accuracy_score(pred['trueTarget'], pred['predTarget']))
		pred = model.getPredict(domain='Source')
		source_metric_i.append(accuracy_score(pred['trueSource'], pred['predSource']))

		model.save_params(save_path,file)
		num_samples.append(ns)
		del model,dm_source,dm_target,trainer
	
	if my_logger:
		log_metr = {}
		log_metr['source acc iter'] = source_metric_i
		log_metr['target acc iter'] = target_metric_i
		log_metr[f'samples selected'] = num_samples
		my_logger.log_metrics(log_metr)
		
	print(log_metr)

