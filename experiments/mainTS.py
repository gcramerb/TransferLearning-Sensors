import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
sys.path.insert(0, '../')
from dataProcessing.dataModule import SingleDatasetModule, CrossDatasetModule
from trainers.runClf import runClassifier
from models.pseudoLabSelection import saveSL,saveSLdim
from trainers.trainerSL import SLmodel
from Utils.myUtils import get_Clfparams, get_SLparams

"""
The main idea of this experiment is to train iterativilly two models, the theacher and the student.
The teacher uses the source and target data with discrepancy loss to learn similar features.
The student are a simple classifer that lerns only by the soft label data from target domain.

"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='Teach_stud_v3')
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
	
	ts_path_file = None
	#ts_path_file = ts_path_file = os.path.join(args.inPath, f'{args.source}_to_{args.target}_pseudo_labels_f25_t2_{args.n_classes}actv.npz')
	first_save = True
	source_metric_i = []
	target_metric_i = []
	num_samplesTea = []
	num_samplesStu = []
	purityTea_i =[]
	purityStu_i = []
	models_name  = ['Teacher','Student']

	file_sl =  f'Model_{args.source}_{args.target}_{models_name[0]}'
	file_clf = f'Model_{args.source}_{args.target}_{models_name[1]}'
	
	if my_logger:
		my_logger.log_hyperparams(clfParams)
		my_logger.log_hyperparams(SLparams)
		
	for i in range(SLparams['iter']):
		dm_source = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.source,
		                                n_classes=args.n_classes,
		                                input_shape=clfParams['input_shape'],
		                                batch_size=SLparams['bs'])
		
		dm_source.setup(normalize = True, SL_path_file=ts_path_file)
	
		# the file name must be in that way because to be readen as "main" data in dataModule
		ts_path_file = os.path.join(args.inPath, f'{args.source}_to_{args.target}_pseudo_labels_f25_t2_{args.n_classes}actv.npz')
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.target,
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=SLparams['bs'])
	
		dm_target.setup(normalize=True)
		model = SLmodel(trainParams=SLparams,
		                n_classes=args.n_classes,
		                lossParams=None,
		                save_path=None,
		                class_weight=class_weight,
		                model_hyp=clfParams)
		model.setDatasets(dm_source, dm_target)
		model.create_model()
		# if my_logger:
		# 	my_logger.watch(model)
	
	
		# if i > 0:
		# 	# pode ser usado para treinar mais rapido (diminui as epocas com early stopping).
		# 	model.load_params(save_path, file_sl)
		# 	first_save = False
	
		early_stopping = EarlyStopping('val_loss', mode='min', patience=10, verbose=True)
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=SLparams['epoch'],
		                  min_epochs=1,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks=[],
		                  multiple_trainloader_mode='max_size_cycle')
	
		trainer.fit(model)
		pred = model.getPredict(domain = 'Target')
		idx = saveSLdim(path_file=ts_path_file, data = pred['dataTarget'],
		            probs = pred['probTarget'],trh = SLparams['trasholdDisc'],
		            first_save = first_save)
		
		softLab = np.argmax(pred['probTarget'][idx], axis=1)
		purityTea_i.append(accuracy_score(pred['trueTarget'][idx], softLab))
		# tem que lembrar o que source esta misturado com o softLabel do target.
		predS = model.getPredict(domain = 'Source')
		accS = accuracy_score(predS['trueSource'], predS['predSource'])
		accT = accuracy_score(pred['trueTarget'], pred['predTarget'])
		source_metric_i.append((accS,accT))
		model.save_params(save_path, file_sl)
		num_samplesTea.append(len(idx))
		print(f'\n\n Iter {i}: Teacher added {len(idx)} samples for SL dataset \n\n')
		print(f'{purityTea_i[-1]} % of those are correct\n')
		print(f'Teacher results at Iter {i}: \n source: {accS}  target: {accT} \n')
		del model, dm_source, trainer,pred,dm_target
		# ----------------------- finished the Teacher part -------------------------------------------#
	
	#for i in range(SLparams['iter']):
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.target,
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=SLparams['bs'])
		
		dm_target.setup(normalize=True)
		dm_SL  = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=  f'{args.source}_to_{args.target}_pseudo_labels',
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=clfParams['bs'])
		dm_SL.setup(normalize=True)
		
		if i > 0:
			trainer, clf = runClassifier(dm_SL, clfParams,my_logger = my_logger,load_params_path =save_path,file = file_clf)
		else:
			trainer, clf = runClassifier(dm_SL, clfParams,my_logger = my_logger)
		
		#trainer, clf = runClassifier(dm_SL, clfParams, my_logger=my_logger)
		# res = metrics = clf.get_all_metrics(dm_SL.test_dataloader())
		# print('Target train (student) in SL data: ', res['test_acc'],'\n')
		
		pred = clf.predict(dm_target.test_dataloader())
		acc = accuracy_score(pred['true'], pred['pred'])
		cm = confusion_matrix(pred['true'],pred['pred'])

		target_metric_i.append(acc)
		stud_metrics = {}
		stud_metrics['Final acc Student'] = acc
		stud_metrics['Final cm Student'] = cm
		

		#TODO: so salvar os pseudo Labels se o treinamento do clf tiver sido bom...
		idx = saveSLdim(path_file=ts_path_file, data = pred['data'],
		            probs = pred['probs'],trh = SLparams['trasholdStu'],
		            first_save = first_save)
		print(f'Student acc in Target data: {acc}')
		pred = clf.predict(dm_SL.test_dataloader())
		acc = accuracy_score(pred['true'], pred['pred'])
		stud_metrics['Student acc in SL'] = acc
		
		if len(idx) > clfParams['bs']:
			softLab = np.argmax(pred['probs'][idx], axis=1)
			purityStu_i.append(accuracy_score(pred['true'][idx], softLab))
			print(f'\n\n Iter {i}: Student added {len(idx)} samples in SL dataset \n')
			print(f'{purityStu_i[-1]} % of those are correct\n')
			first_save = False
		num_samplesStu.append(len(idx))
		clf.save_params(save_path, file_clf)
		del clf, trainer,dm_SL,dm_target
		
	if my_logger:
		log_metr = {}
		log_metr['source acc iter (teacher)'] = source_metric_i
		log_metr['target acc iter (student)'] = target_metric_i
		log_metr[f'samples selected by Teacher'] = num_samplesTea
		log_metr[f'samples selected by Student'] = num_samplesStu
		log_metr['SL purity inter (teacher)'] = purityTea_i
		log_metr['SL purity inter (student)'] = purityStu_i
		my_logger.log_metrics(log_metr)
		my_logger.log_metrics(stud_metrics)
	print(stud_metrics)