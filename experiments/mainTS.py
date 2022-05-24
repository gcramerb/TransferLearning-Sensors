import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
sys.path.insert(0, '../')
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import saveSL,saveSLdim,saveSL_exp
from trainers.trainerTL import TLmodel
from trainers.trainerClf import ClfModel
from Utils.myUtils import get_Clfparams, get_SLparams, MCI

"""
The main idea of this experiment is to train iterativilly two models, the theacher and the student.
The teacher uses the source and target data with discrepancy loss to learn similar features.
The student are a simple classifer that lerns only by the soft label data from target domain.

"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='Stu_bench')
parser.add_argument('--SLParamsFile', type=str, default=None)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Uschad")
parser.add_argument('--target', type=str, default="Dsads")
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
	# my_logger = WandbLogger(project='TransferLearning-Soft-Label',
	#                         log_model='all',
	#                         name=args.expName + args.source + '_to_' + args.target)
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'



def runTS(clfParams,SLparams,expName):
	metrics = {}
	metrics['Student acc in Target'] = []
	metrics['Student acc in SL'] = []
	metrics[f'Num samples selected'] = []
	metrics['SL purity by Student'] = []
	
	class_weight = None
	
	SLdatasetName = f'{args.source}_to_{args.target}_PS_{expName}'
	ts_path_file = os.path.join(args.inPath, f'{SLdatasetName}_f25_t2_{args.n_classes}actv.npz')
	first_save = True

	file = f'DiscSaved_{args.source}_{args.target}'
	file_clf = f'Student_{args.source}_{args.target}'
	
	if my_logger:
		my_logger.log_hyperparams(SLparams)
	# my_logger.log_hyperparams(clfParams)

	for i in range(SLparams['iter']):
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.target,
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=SLparams['bs'])
		
		dm_target.setup(normalize=True)
		clf = ClfModel(lr=clfParams['clf_lr'],
		               n_classes=args.n_classes,
		               alpha=clfParams['alpha'],
		               step_size=clfParams['step_size'],
		               model_hyp=clfParams,
		               weight_decay=clfParams['weight_decay'],
		               class_weight=class_weight,
		               input_shape=clfParams['input_shape'])
		
		if my_logger:
			adicionalInfo = {}
			adicionalInfo['class_weight'] = class_weight
			my_logger.log_hyperparams(adicionalInfo)
			my_logger.watch(clf, log_graph=False)
		
		clf.load_params(save_path, file)
		file = file_clf
		pred = clf.predict(dm_target.test_dataloader())
		acc = accuracy_score(pred['true'], pred['pred'])
		print(f'Student acc in Target data: {acc}')
		metrics['Student acc in Target'].append(acc)
		
		new_idx = saveSL(path_file=ts_path_file, data=pred['data'],
		                     probs=pred['probs'], trh=SLparams['trasholdStu'],
		                     first_save=first_save)
		
		if len(new_idx) > clfParams['bs']:
			softLab = np.argmax(pred['probs'][new_idx], axis=1)
			pu = accuracy_score(pred['true'][new_idx], softLab)
			metrics['SL purity by Student'].append(pu)
			print(f'\n\n Iter {i}: Student added {len(new_idx)} samples in SL dataset \n')
			print(f'{pu * 100} % of those are correct\n')
			first_save = False
		
		metrics[f'Num samples selected'].append(len(new_idx))
		
		# TODO: so salvar os pseudo Labels se o treinamento do clf tiver sido bom...
		dm_SL = SingleDatasetModule(data_dir=args.inPath,
		                            datasetName=SLdatasetName,
		                            input_shape=clfParams['input_shape'],
		                            n_classes=args.n_classes,
		                            batch_size=clfParams['bs'])
		dm_SL.setup(normalize=True)
		early_stopping = EarlyStopping('val_loss', mode='min', min_delta=0.05, patience=4, verbose=True)
		trainer = Trainer(gpus=1,
		                  logger=my_logger,
		                  check_val_every_n_epoch=1,
		                  max_epochs=clfParams['clf_epoch'],
		                  progress_bar_refresh_rate=0,
		                  callbacks=[early_stopping])
		
		trainer.fit(clf, datamodule=dm_SL)
		clf.save_params(save_path, file_clf)
		predSL = clf.predict(dm_SL.test_dataloader())
		acc = accuracy_score(predSL['true'], predSL['pred'])
		print(f'Student acc in SL data: {acc}')
		metrics['Student acc in SL'].append(acc)
		del clf, trainer, dm_SL,dm_target
	
	cm = confusion_matrix(pred['true'], pred['pred'])
	metrics['Final cm Student'] = cm
	return metrics


if __name__ == '__main__':
	path_clf_params, path_SL_params,class_weight = None, None,None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path, args.ClfParamsFile)
	if args.SLParamsFile:
		path_SL_params = os.path.join(params_path, args.SLParamsFile)
	
	clfParams = get_Clfparams(path_clf_params)
	SLparams = get_SLparams(path_SL_params)
	metrics = runTS(clfParams,SLparams,args.expName)
	if my_logger:
		my_logger.log_metrics(metrics)
	print(metrics)