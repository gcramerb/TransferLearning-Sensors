import sys, argparse, os, glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerTL import TLmodel
from Utils.myUtils import MCI, getTeacherParams

# seed = 2804
# print('Seeding with {}'.format(seed))
# torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--expName', type=str, default='__')
parser.add_argument('--TLParamsFile', type=str, default=None)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--freq', type=int, default=100)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--model', type=str, default="V5")
parser.add_argument('--savePath', type=str, default=None)
args = parser.parse_args()

my_logger = None
paramsPath = None
if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/{args.model}/'
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\originalWindFreq\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\{args.model}\\'
	args.savedPath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\teacherOficialV5\\'

	args.log = False

if args.TLParamsFile:
	paramsPath = os.path.join(params_path, args.TLParamsFile)
else:
	paramsPath = os.path.join(params_path, "Disc" + args.source[:3] + args.target[:3] + ".json")
if args.log:
	my_logger = WandbLogger(project='TransferLearning-Soft-Label',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)

def calculateMetrics(pred,true):
	final_result = {}
	final_result["Acc"] = []
	final_result["F1"] = []
	for class_ in range(4):
		final_result[f"Acc class {class_}"] = []
	acc = accuracy_score(pred,true)
	f1 = f1_score(true, pred, average='weighted')
	cm = confusion_matrix(true, pred)
	# for class_ in range(4):
	#
	# 	final_result[f"Acc class {class_}"] = cm[class_][class_] / cm[class_][:].sum()
	return acc,final_result
	
def runDisc(teacherParams, dm_source, dm_target, trials, save_path=None, useMixup=True):
	best_acc = 0
	dictMetricsAll = []
	for i in range(trials):
		teacherParams['input_shape'] = dm_source.dataTrain.X.shape[1:]
		model = TLmodel(trainParams=teacherParams,
		                n_classes=args.n_classes,
		                lossParams=None,
		                useMixup=useMixup,
		                save_path=None,
		                class_weight=torch.tensor([0.5, 2, 2, 0.5]))
		
		model.setDatasets(dm_source, dm_target)
		model.create_model()
		from torchsummary import summary
		#summary(model.to("cuda").FE, teacherParams['input_shape'])
		#model.load_params(args.savedPath, f'Teacher{args.model}_{args.source}_{args.target}')

		#early_stopping = EarlyStopping('valAccTarget', mode='max', patience=10, verbose=True)
		trainer = Trainer(devices=1,
		                  accelerator="gpu",
		                  check_val_every_n_epoch=1,
		                  max_epochs=teacherParams['epoch'],
		                  logger=my_logger,
		                  enable_progress_bar=False,
		                  min_epochs=1,
		                  callbacks=[],
		                  enable_model_summary=True,
		                  multiple_trainloader_mode='max_size_cycle')
		
		if my_logger:
			my_logger.watch(model, log_graph=False)
		trainer.fit(model)
		predT = model.getPredict(domain='Target')
		predS = model.getPredict(domain='Source')
		accT,dictMetricsT = calculateMetrics(predT['trueTarget'], predT['predTarget'])
		dictMetricsAll.append(dictMetricsT)
		if accT > best_acc:
			best_acc = accT
			if save_path is not None   :
				print(f"saving: {dm_source.datasetName} to {dm_target.datasetName} with Acc {accT}\n\n")
				print(teacherParams)
				model.save_params(save_path, f'Teacher{args.model}_{args.source}_{args.target}')
				
		del model, trainer
	print(f'\n-------------------------------------------------------\n BEST Acc target {best_acc}\n')
	print('-----------------------------------------------------------')
	return best_acc, dictMetricsAll


if __name__ == '__main__':
	print(f"params loaded from: {paramsPath}")
	useMixup = False
	teacherParams = getTeacherParams(paramsPath)
	
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName="",
	                                n_classes=args.n_classes,
	                                freq=args.freq,
	                                input_shape=2,
	                                batch_size=128,
	                                oneHotLabel=useMixup,
	                                shuffle=True)

	filename = f"{args.source}AllOriginal_target_{args.target}AllOriginal.npz"
	dm_source.setup(normalize=False,fileName =filename )
	#dm_source.setup(normalize=True, fileName=f"{args.source}AllOriginal.npz")

	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName="",
	                                input_shape=2,
	                                freq=args.freq,
	                                n_classes=args.n_classes,
	                                batch_size=128,
	                                oneHotLabel=useMixup,
	                                shuffle=True)
	dm_target.setup(normalize=False,fileName = f"{args.target}AllOriginal_target_{args.source}AllOriginal.npz")
	#dm_target.setup(normalize=True, fileName=f"{args.target}AllOriginal.npz")
	
	final_result = runDisc(teacherParams, dm_source, dm_target, args.trials, args.savePath, useMixup=useMixup)
	print(final_result)
	if my_logger:
		my_logger.log_metrics(final_result)
