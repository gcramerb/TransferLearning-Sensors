import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, '../')
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from trainers.trainerTL import TLmodel
from Utils.myUtils import MCI, getTeacherParams, getStudentParams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='student')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--studentPath', action='store_true')
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="V4")
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--freq', type=int, default=50)
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
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
teacherParamsPath = os.path.join(params_path, "Disc" + args.source[:3] + args.target[:3] + ".json")

if args.log:
	my_logger = WandbLogger(project='studentOficial',
	                        log_model='all',
	                        name=args.expName + f'{args.model}' + args.source + '_to_' + args.target)


def runStudent(studentParams, source, target, class_weight=None, my_logger=None,trials = 1, pre_train = False):
	final_result = {}
	final_result["Acc Target"] = []
	final_result["F1 Target"] = []
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = []
	for i in range(trials):
		batchSize = 64

		model = ClfModel(trainParams=studentParams,
		                 class_weight=class_weight,
		                 oneHotLabel=False,
		                 mixup = False)
		model.create_model()
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName=target,
		                                input_shape=(2, args.freq * 2, 3),
		                                freq=args.freq,
		                                n_classes=args.n_classes,
		                                batch_size=batchSize,
		                                oneHotLabel=True,
		                                shuffle=True)
		
		dm_target.setup(normalize=True)
		#early_stopping = EarlyStopping('training_loss', mode='min', patience=10, verbose=True)
		early_stopping = []

		trainer = Trainer(devices=1,
		                  accelerator="gpu",
		                  check_val_every_n_epoch=1,
		                  max_epochs=15,
		                  logger=my_logger,
		                  enable_progress_bar=False,
		                  min_epochs=1,
		                  callbacks=[checkpoint_callback],
		                  enable_model_summary=True)
		if pre_train:
			dm_source = SingleDatasetModule(data_dir=args.inPath,
			                                datasetName=source,
			                                input_shape=(2, args.freq * 2, 3),
			                                freq=args.freq,
			                                n_classes=args.n_classes,
			                                batch_size=batchSize,
			                                oneHotLabel=False,
			                                shuffle=True)
			
			dm_source.setup(normalize=True)
			model.setDatasets(dm=dm_source,secondDataModule=dm_target)
			trainer.fit(model)
		dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
		                                     datasetName=f"",
		                                     input_shape=(2, args.freq * 2, 3),
		                                     freq=args.freq,
		                                     n_classes=args.n_classes,
		                                     batch_size=batchSize,
		                                     oneHotLabel=False,
		                                     shuffle=True)
		
		fileName = f"{args.source}_{args.target}pseudoLabel{args.model}.npz"
		dm_pseudoLabel.setup(normalize=True, fileName=fileName)
		model.setDatasets(dm=dm_pseudoLabel, secondDataModule=dm_target)
		trainer.fit(model,ckpt_path='/teste.ckpt')
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
	studentParams = getStudentParams()
	studentParams['input_shape'] = (2, args.freq * 2, 3)
	class_weight = torch.tensor([0.5, 2, 2, 0.5])
	studentParams['class_weight'] = class_weight
	pre_train = True
	metrics = runStudent(studentParams, args.source, args.target, class_weight=class_weight, my_logger=my_logger,trials = args.trials,pre_train = pre_train)
	print(metrics)
	if my_logger:
		my_logger.log_hyperparams(studentParams)
		my_logger.log_metrics(metrics)
