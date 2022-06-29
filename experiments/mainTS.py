import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
sys.path.insert(0, '../')
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import saveSL,saveSLdim,saveSL_gmm
from trainers.trainerTL import TLmodel
from trainers.trainerClf import ClfModel
from Utils.myUtils import get_Clfparams, get_Stuparams, MCI

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


def generateFirstPL(tsPathFile,teacherParams,source,target):
	
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=teacherParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=teacherParams['bs'])
	
	dm_target.setup(normalize=True)
	teacher = ClfModel(lr=teacherParams['lr'],
	                   n_classes=args.n_classes,
	                   alpha=teacherParams['alpha'],
	                   step_size=teacherParams['step_size'],
	                   model_hyp=teacherParams,
	                   weight_decay=teacherParams['weight_decay'],
	                   class_weight=class_weight,
	                   input_shape=teacherParams['input_shape'])
	
	file = f'DiscSaved_{source}_{target}'
	teacher.load_params(save_path, file)
	pred = teacher.predict(dm_target.test_dataloader())
	acc = accuracy_score(pred['true'], pred['pred'])
	print(f'Init (Teacher) acc in Target{acc}')

	#save all samples:
	saveAllLabels(path_file=tsPathFile, data=pred['data'],
	                 yTrue = pred['true'],probs=pred['probs'],pred['latent'])
	del teacher, dm_target
	return True



def runTS(teacherParams,stuParams,expName,source,target):
	metrics = {}
	metrics['Student acc in Target'] = []
	metrics['Student acc in SL'] = []
	# metrics[f'Num samples selected'] = []
	# metrics['SL purity by Student'] = []
	
	class_weight = None
	
	tsDatasetName = f'{source}_to_{target}_PS_{expName}'
	tsPathFile = os.path.join(args.inPath, f'{tsDatasetName}_f25_t2_{args.n_classes}actv.npz')
	
	if not os.path.isfile(tsPathFile):
		generateFirstPL(tsPathFile,teacherParams, source, target)

	first_save = True
	studentDataset = f'Student_{source}_{target}'
	studentPathFile = os.path.join(args.inPath, f'{student_dataset}_f25_t2_{args.n_classes}actv.npz')
	
	if my_logger:
		adicionalInfo = {}
		adicionalInfo['class_weight'] = class_weight
		my_logger.log_hyperparams(adicionalInfo)
		my_logger.log_hyperparams(stuParams)
	
	#for i in range(stuParams['iter']):
		
	
	simplest_SLselec(tsPathFile, studentPathFile, 0.75)

	# TODO: so salvar os pseudo Labels se o treinamento do student tiver sido bom...
	studentDm = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=studentDataset,
	                                input_shape=stuParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=stuParams['bs'])
	studentDm.setup(normalize=True)
	
	teacher = ClfModel(lr=stuParams['lr'],
	                   n_classes=args.n_classes,
	                   alpha=stuParams['alpha'],
	                   step_size=stuParams['step_size'],
	                   model_hyp=stuParams,
	                   weight_decay=stuParams['weight_decay'],
	                   class_weight=class_weight,
	                   input_shape=stuParams['input_shape'])
	
	early_stopping = EarlyStopping('val_loss', mode='min', min_delta=0.05, patience=4, verbose=True)
	trainer = Trainer(gpus=1,
	                  logger=my_logger,
	                  check_val_every_n_epoch=1,
	                  max_epochs=stuParams['clf_epoch'],
	                  progress_bar_refresh_rate=0,
	                  callbacks=[early_stopping])
	
	trainer.fit(student, datamodule=studentDm)
	
	#student.save_params(save_path, file_clf)
	predSL = student.predict(studentDm.test_dataloader())
	acc = accuracy_score(predSL['true'], predSL['pred'])
	print(f'Student acc in SL data: {acc}')
	metrics['Student acc in SL'].append(acc)
	
	if my_logger:
		my_logger.watch(student, log_graph=False)
	
	# if len(new_idx) > stuParams['bs']:
	# 	softLab = np.argmax(pred['probs'][new_idx], axis=1)
	# 	pu = accuracy_score(pred['true'][new_idx], softLab)
	# 	metrics['SL purity by Student'].append(pu)
	# 	print(f'\n\n Iter {i}: Student added {len(new_idx)} samples in SL dataset \n')
	# 	print(f'{pu * 100} % of those are correct\n')
	# 	first_save = False
	# metrics[f'Num samples selected'].append(len(new_idx))
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=teacherParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=teacherParams['bs'])
	
	dm_target.setup(normalize=True)
	predStu = student.predict(studentDm.test_dataloader())
	acc = accuracy_score(predStu['true'], predStu['pred'])
	print(f'\n Student acc in Target data: {acc}')
	metrics['Student acc in Target'].append(acc)
	cm = confusion_matrix(predStu['true'], predStu['pred'])
	metrics['Final cm Student'] = cm
	del student, trainer, studentDm,dm_target
	return metrics


if __name__ == '__main__':
	path_Stu_params,class_weight = None, None,None
	if args.StuParamsFile:
		path_Stu_params = os.path.join(params_path, args.StuParamsFile)
	path_teacher_params = os.path.join(params_path, args.teacherParamsFile)
	teacherParams = get_TLparams(path_teacher_params)
	stuParams = get_Stuparams(path_Stu_params)
	metrics = runTS(teacherParams,stuParams,"teacher_student",args.source,args.target)
	if my_logger:
		my_logger.log_metrics(metrics)
	print(metrics)