import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from pytorch_lightning.loggers import WandbLogger
sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from trainers.trainerTL import TLmodel
from Utils.myUtils import  MCI,getTeacherParams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='mainStudent')
parser.add_argument('--TLParamsFile', type=str, default="DiscUscDsa.json")
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Pamap2")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--saveModel', type=bool, default=False)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.savePath  = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\testsTeacher\\'
if args.log:
	my_logger = WandbLogger(project='studentOficial',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)

def savePseudoLabels(teacherParams,source, target):
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=(2, 50, 3),
	                                n_classes=args.n_classes,
	                                batch_size=128,
	                                shuffle=True)
	
	dm_target.setup(normalize=True)
	model = TLmodel(trainParams=teacherParams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                save_path=None,
	                class_weight=None)
	
	model.setDatasets(dm_target=dm_target)
	model.create_model()
	model.load_params(args.savePath, f'Disc_class{source}_{target}')
	predT = model.getPredict(domain='Target')
	
	Xpl, yPl, _ = getPseudoLabel(pred.copy(), method='cluster', param=64)
	fileName = f"{source}_{target}pseudoLabel.npz"
	path_file = os.path.join(args.inPath,fileName)
	with open(path_file, "wb") as f:
		np.savez(f, X=Xpl,y = yPl folds=np.zeros(1))
	del model,dm_target
	dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=f"pseudoLabel_{source}_{target}",
	                                input_shape=(2, 50, 3),
	                                n_classes=args.n_classes,
	                                batch_size=128,
	                                shuffle=True)
	
	dm_pseudoLabel.setup(normalize=True,fileName = fileName)
	model = ClfModel(trainParams=teacherParams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                save_path=None,
	                class_weight=None)
	
	model.setDatasets(dm_source=dm_pseudoLabel)
	model.create_model()
	early_stopping = EarlyStopping('loss', mode='min', patience=10, verbose=True)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=teacherParams['epoch'],
	                  logger=my_logger,
	                  min_epochs=1,
	                  progress_bar_refresh_rate=verbose,
	                  callbacks=[early_stopping],
	                  enable_model_summary=True,
	                  multiple_trainloader_mode='max_size_cycle')
	trainer.fit(model)
	predT = model.getPredict(domain='Source')
		
	




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
	SLparams = get_Stuparams(path_SL_params)
	if my_logger:
		my_logger.log_hyperparams(clfParams)
		my_logger.log_hyperparams(SLparams)

	first_save = True
	sl_path_file = None
	source_metric_i = []
	target_metric_i = []
	puritySL_i =[]
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
		model = TLmodel(trainParams=SLparams,
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
		# early_stopping = EarlyStopping('val_acc_target', mode='max', patience=10, verbose=True)
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=SLparams['epoch'],
		                  min_epochs=1,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks=[],
		                  multiple_trainloader_mode='max_size_cycle')
		
		trainer.fit(model)
		sl_path_file = os.path.join(args.inPath, f'{args.target}_to_{args.n_classes}_mainSL.npz')
		pred = model.getPredict(domain='Target')
		idx = saveSL(path_file=sl_path_file, data = pred['dataTarget'],
		            probs = pred['probTarget'],trh = SLparams['trasholdDisc'],
		            first_save = first_save)
		target_metric_i.append(accuracy_score(pred['trueTarget'], pred['predTarget']))
		softLab = np.argmax(pred['probTarget'][idx], axis=1)
		puritySL_i.append(accuracy_score(pred['trueTarget'][idx], softLab))
		print(f'\n\n Iter {i}: added {len(idx)} samples for SL dataset \n\n')
		print(f'{puritySL_i[-1]} % of those are correct\n')
		
		pred = model.getPredict(domain='Source')
		source_metric_i.append(accuracy_score(pred['trueSource'], pred['predSource']))

		model.save_params(save_path,file)
		num_samples.append(len(idx))
		del model,dm_source,dm_target,trainer
		first_save = False
	
	if my_logger:
		log_metr = {}
		log_metr['source acc iter'] = source_metric_i
		log_metr['target acc iter'] = target_metric_i
		log_metr['target purity'] = puritySL_i
		log_metr[f'samples selected'] = num_samples
		my_logger.log_metrics(log_metr)
		
	print(log_metr)

