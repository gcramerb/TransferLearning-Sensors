import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from pytorch_lightning.loggers import WandbLogger
sys.path.insert(0, '../')
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from trainers.trainerTL import TLmodel
from Utils.myUtils import  MCI,getTeacherParams,getStudentParams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='mainStudent')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--savePath', type=str, default='../saved/hypDisc/')
parser.add_argument('--studentPath', action='store_true')
parser.add_argument('--source', type=str, default="Ucihar")
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
	paramsPath = os.path.join(params_path, args.source[:3] + args.target[:3] + ".json")

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
	args.savePath  = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\testsTeacher\\'
	paramsPath = os.path.join(params_path, args.source[:3] + args.target[:3] + ".json")
if args.studentPath:
	studentPath = os.path.join(params_path, "student.json")
else:
	studentPath = None
if args.log:
	my_logger = WandbLogger(project='studentOficial',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)


def runStudent(studentParams,source,target,class_weight = None,my_logger = None):
	batchSize = 64
	dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=f"pseudoLabel_{source}_{target}",
	                                input_shape=(2, 50, 3),
	                                n_classes=args.n_classes,
	                                batch_size=batchSize,
	                                shuffle=True)
	
	fileName = f"{source}_{target}pseudoLabel.npz"
	path_file = os.path.join(args.inPath,fileName)
	dm_pseudoLabel.setup(normalize=True,fileName = fileName)
	model = ClfModel(trainParams=studentParams,
	                class_weight=class_weight)
	
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=(2, 50, 3),
	                                n_classes=args.n_classes,
	                                batch_size=batchSize,
	                                shuffle=True)
	
	dm_target.setup(normalize=True)
	model.setDatasets(dm=dm_pseudoLabel,secondDataModule = dm_target)
	model.create_model()
	early_stopping = EarlyStopping('training_loss', mode='min', patience=5, verbose=True)
	log  = int(dm_target.X_train.__len__()/batchSize)
	
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=studentParams['epoch'],
	                  logger=my_logger,
	                  min_epochs=1,
	                  log_every_n_steps=log,
	                  progress_bar_refresh_rate=0,
	                  callbacks=[early_stopping],
	                  enable_model_summary=True)
	trainer.fit(model)

	pred = model.predict(dm_target.test_dataloader())
	final_result = {}
	final_result["Acc Target"] = accuracy_score(pred['true'], pred['pred'])
	final_result["CM Target"] = confusion_matrix(pred['true'], pred['pred'])
	final_result['F1 Target'] = f1_score(pred['true'], pred['pred'],average = 'weighted')
	final_result['len target on predict'] = len(pred['pred'])
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = final_result["CM Target"][class_][class_]/final_result["CM Target"][class_][:].sum()
		
	return final_result


if __name__ == '__main__':
	
	
	studentParams = getStudentParams(studentPath)
	if args.source == 'Uschad':
		class_weight = torch.tensor([0.5, 5, 5, 0.5])
	else:
		class_weight = None
	studentParams['class_weight'] = class_weight
	metrics = runStudent(studentParams,args.source,args.target,class_weight = class_weight,my_logger = my_logger)

	print(metrics)
	if my_logger:
		my_logger.log_hyperparams(studentParams)
		my_logger.log_metrics(metrics)
