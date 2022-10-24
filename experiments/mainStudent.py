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
from Utils.myUtils import MCI, getTeacherParams, getStudentParams

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
parser.add_argument('--trials', type=int, default=1)
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
	args.savePath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\testsTeacher\\'
	paramsPath = os.path.join(params_path, args.source[:3] + args.target[:3] + ".json")
if args.studentPath:
	studentPath = os.path.join(params_path, "student.json")
else:
	studentPath = None
if args.log:
	my_logger = WandbLogger(project='studentOficial',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)


def runStudent(studentParams, source, target, class_weight=None, my_logger=None,trials = 1):
	final_result = {}
	final_result["Acc Target"] = []
	final_result["F1 Target"] = []
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = []
	for i in range(trials):
		batchSize = 64
		dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
		                                     datasetName=f"pseudoLabel_{source}_{target}2steps",
		                                     input_shape=(2, 50, 3),
		                                     n_classes=4,
		                                     batch_size=batchSize,
		                                     oneHotLabel=True,
		                                     shuffle=True)
		
		fileName = f"{source}_{target}pseudoLabel.npz"
		path_file = os.path.join(args.inPath, fileName)
		dm_pseudoLabel.setup(normalize=True, fileName=fileName)
		model = ClfModel(trainParams=studentParams,
		                 class_weight=class_weight)
		
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=target,
		                                input_shape=(2, 50, 3),
		                                n_classes=4,
		                                batch_size=batchSize,
		                                oneHotLabel=True,
		                                shuffle=True)
		
		dm_target.setup(normalize=True)
		model.setDatasets(dm=dm_pseudoLabel, secondDataModule=dm_target)
		model.create_model()
		early_stopping = EarlyStopping('training_loss', mode='min', patience=10, verbose=True)
		log = int(dm_target.X_train.__len__() / batchSize)
		
		trainer = Trainer(gpus=1,
		                  check_val_every_n_epoch=1,
		                  max_epochs=studentParams['epoch'],
		                  logger=my_logger,
		                  enable_progress_bar=False,
		                  min_epochs=1,
		                  log_every_n_steps=log,
		                  callbacks=[early_stopping],
		                  enable_model_summary=True)
		trainer.fit(model)
		
		pred = model.predict(dm_target.test_dataloader())

		final_result["Acc Target"].append(accuracy_score(pred['true'], pred['pred']))
		cm = confusion_matrix(pred['true'], pred['pred'])
		final_result['F1 Target'].append(f1_score(pred['true'], pred['pred'], average='weighted'))
		for class_ in range(4):
			final_result[f"Acc Target class {class_}"].append(cm[class_][class_] / \
			                                             cm[class_][:].sum())
	final_result['Target acc mean'] = MCI(final_result["Acc Target"])
	final_result['Target f1 mean'] =  MCI(final_result["F1 Target"])
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = MCI(final_result[f"Acc Target class {class_}"])
	return final_result


if __name__ == '__main__':
	studentParams = getStudentParams(studentPath)
	class_weight = torch.tensor([0.5, 5, 5, 0.5])
	if args.source == 'Uschad':
		class_weight = torch.tensor([0.5, 10, 10, 0.5])
	
	studentParams['class_weight'] = class_weight
	metrics = runStudent(studentParams, args.source, args.target, class_weight=class_weight, my_logger=my_logger,trials = args.trials)
	
	print(metrics)
	if my_logger:
		my_logger.log_hyperparams(studentParams)
		my_logger.log_metrics(metrics)
