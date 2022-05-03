import sys, argparse, os, glob

sys.path.insert(0, '../')
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule, CrossDatasetModule
from trainers.runClf import runClassifier
from models.pseudoLabSelection import saveSL
from trainers.trainerSL import SLmodel
from Utils.myUtils import get_Clfparams, get_TLparams, get_SLparams

"""
The main idea of this experiment is to train iterativilly two models, the theacher and the student.
The teacher uses the source and target data with discrepancy loss to learn similar features.
The student are a simple classifer that lerns only by the soft label data from target domain.

"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='Teach_stud')
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
	
	path_clf_params, path_TL_params = None, None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path, args.ClfParamsFile)
	if args.SLParamsFile:
		path_TL_params = os.path.join(params_path, args.SLParamsFile)
	
	if args.source == 'Uschad':
		class_weight = torch.tensor([0.5, 5, 5, 0.5])
	else:
		class_weight = None
	
	clfParams = get_Clfparams(path_clf_params)
	SLparams = get_SLparams(path_TL_params)
	
	ts_path_file = None
	#ts_path_file = ts_path_file = os.path.join(args.inPath, f'{args.source}_to_{args.target}_pseudo_labels_f25_t2_{args.n_classes}actv.npz')
	first_save = True
	source_metric_i = []
	target_metric_i = []
	num_samplesT = []
	num_samplesS = []
	models_name  = ['teacher','student']

	file_clf = f'Model_{args.source}_{args.target}_Student'
	file_sl =  f'Model_{args.source}_{args.target}_Teacher'
	
	if my_logger:
		my_logger.log_hyperparams(clfParams)
		my_logger.log_hyperparams(SLparams)

	for i in range(SLparams['iter']):
		dm_source = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=args.source,
		                                n_classes=args.n_classes,
		                                input_shape=clfParams['input_shape'],
		                                batch_size=clfParams['bs'])
		
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
		if my_logger:
			my_logger.watch(model)

		#TODO: It is reallly necessary to save the params? Why I did that?
		if i > 0:
		# 	model.load_params(save_path, file_sl)
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
		pred = model.getPredict(domain = 'Target')
		ns = saveSL(path_file=ts_path_file, data = pred['dataTarget'],
		            probs = pred['probTarget'],trh = SLparams['trasholdDisc'],
		            first_save = first_save)
		print(f'\n\n\n Iter {i}: SL len = {ns}\n\n\n')
		del pred
		#ns = model.save_pseudoLab(path_file=ts_path_file,first_save = first_save)

		# TODO: It is reallly necessary to save the params? Why I did that?
		#model.save_params(save_path, file_sl)

		out = model.get_final_metrics()
		source_metric_i.append(out['acc_source_all'])
		target_metric_i.append(out['acc_target_all'])
		print(f'Results at Iter {i}  - \n {out} \n')
		num_samplesT.append(ns)
		del model, dm_source, trainer

		dm_SL  = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=  f'{args.source}_to_{args.target}_pseudo_labels',
		                                input_shape=clfParams['input_shape'],
		                                n_classes=args.n_classes,
		                                batch_size=SLparams['bs'])
		
		dm_SL.setup(normalize=True)
		
		if i > 0:
			trainer, clf, res = runClassifier(dm_SL, clfParams,my_logger = my_logger,load_params_path =save_path,file = file_clf)
			
		else:
			trainer, clf, res = runClassifier(dm_SL, clfParams,my_logger = my_logger)

		print('Target (first train): ', res['train_acc'])
		predictions = clf.predict(dm_target.test_dataloader())

		ns = saveSL(path_file=ts_path_file, data = predictions['data'],
		            probs = predictions['probs'],trh = SLparams['trasholdStu'],
		            first_save = first_save)
		num_samplesS.append(ns)
		clf.save_params(save_path, file_clf)
		del clf, trainer

	if my_logger:
		log_metr = {}
		log_metr['source acc iter'] = source_metric_i
		log_metr['target acc iter'] = target_metric_i
		log_metr[f'samples selected Teacher'] = num_samplesT
		log_metr[f'samples selected Student'] = num_samplesS
		my_logger.log_metrics(log_metr)

	
	# evaluating the model:
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                input_shape=clfParams['input_shape'],
	                                batch_size=clfParams['bs'])
	dm_source.setup(normalize=True)
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
	                class_weight=None,
	                model_hyp=clfParams)
	
	model.setDatasets(dm_source, dm_target)
	model.create_model()
	model.load_params(save_path, file_sl)
	outcomes = model.get_final_metrics()
	print("final Results (teacher): \n",outcomes,'\n\n')
	
	trainer, clf, res = runClassifier(dm_target, clfParams, load_params_path=save_path, file=file_clf)
	
	print("final results (student): ",res,'\n')
	if my_logger:
		my_logger.log_metrics(outcomes)
		my_logger.log_metrics(res)
		my_logger.log_hyperparams(SLparams)