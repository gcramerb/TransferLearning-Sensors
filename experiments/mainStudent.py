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

	pred = {}
	pred['latent'] = predT['latentTarget']
	pred['pred'] = predT['predTarget']
	pred['true'] = predT['trueTarget']
	pred['probs'] = predT['probTarget']
	pred['data'] = predT['dataTarget']
	
	Xpl, yPl, _ = getPseudoLabel(pred.copy(), method='cluster', param=64)
	fileName = f"{source}_{target}pseudoLabel.npz"
	path_file = os.path.join(args.inPath,fileName)
	with open(path_file, "wb") as f:
		np.savez(f, X=Xpl,y = yPl,folds=np.zeros(1))
	del model,dm_target

def runStudent(source,target,class_weight = None,my_logger = None):
	dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=f"pseudoLabel_{source}_{target}",
	                                input_shape=(2, 50, 3),
	                                n_classes=args.n_classes,
	                                batch_size=64,
	                                shuffle=False)
	
	fileName = f"{source}_{target}pseudoLabel.npz"
	path_file = os.path.join(args.inPath,fileName)
	dm_pseudoLabel.setup(normalize=True,fileName = fileName)
	model = ClfModel(trainParams=studentParams,
	                class_weight=class_weight)
	
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=(2, 50, 3),
	                                n_classes=args.n_classes,
	                                batch_size=64,
	                                shuffle=False)
	
	dm_target.setup(normalize=True)
	model.setDatasets(dm=dm_pseudoLabel,secondDataModule = dm_target)
	model.create_model()
	#early_stopping = EarlyStopping('loss', mode='min', patience=10, verbose=True)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=studentParams['epoch'],
	                  logger=my_logger,
	                  min_epochs=1,
	                  progress_bar_refresh_rate=0,
	                  callbacks=[],
	                  enable_model_summary=True,
	                  multiple_trainloader_mode='max_size_cycle')
	trainer.fit(model)

	pred = model.predict(dm_target.test_dataloader())
	final_result = {}
	final_result["Acc Target"] = accuracy_score(pred['true'], pred['pred'])
	final_result["CM Target"] = confusion_matrix(pred['true'], pred['pred'])
	final_result['F1 Target'] = f1_score(pred['true'], pred['pred'],average = 'weighted')
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = final_result["CM Target"][class_][class_]/final_result["CM Target"][class_][:].sum()
		
	return final_result


if __name__ == '__main__':
	
	teacherParams = getTeacherParams(paramsPath)
	studentParams = getStudentParams()
	if args.source == 'Uschad':
		class_weight = torch.tensor([0.5, 5, 5, 0.5])
		savePseudoLabels(teacherParams, args.source, args.target)
	else:
		class_weight = None
	metrics = runStudent(args.source,args.target,class_weight = class_weight,my_logger = my_logger)
	print(metrics)
	if my_logger:
		my_logger.log_hyperparams(studentParams)
		my_logger.log_metrics(metrics)
