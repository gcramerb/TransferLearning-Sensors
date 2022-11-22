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
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--freq', type=int, default=100)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--savePath', type=str, default=None)
args = parser.parse_args()

my_logger = None
paramsPath = None
if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/V4/'


else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Google Drive\\Mestrado\\Transfer Learning\\Best Results\\V4'
	# args.savePath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\testsTeacher\\'
	savePath = False

if args.TLParamsFile:
	paramsPath = os.path.join(params_path, args.TLParamsFile)
else:
	paramsPath = os.path.join(params_path, "Disc" + args.source[:3] + args.target[:3] + ".json")
if args.log:
	my_logger = WandbLogger(project='Disc',
	                        log_model='all',
	                        name=args.expName + args.source + '_to_' + args.target)


def runDisc(teacherParams, dm_source, dm_target, trials, save_path=None, useMixup=True):
	final_result = {}
	final_result["Acc Target"] = []
	final_result["Acc Source"] = []
	final_result["F1 Source"] = []
	final_result["F1 Target"] = []
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = []
		final_result[f"Acc Source class {class_}"] = []
	best_acc = 0
	for i in range(trials):
		
		model = TLmodel(trainParams=teacherParams,
		                n_classes=args.n_classes,
		                lossParams=None,
		                useMixup=useMixup,
		                save_path=None,
		                class_weight=torch.tensor([0.5, 2, 2, 0.5]))
		
		model.setDatasets(dm_source, dm_target)
		model.create_model()
		#
		from torchsummary import summary
		summary(model.to("cuda").FE, (2, 100, 3))
		
		early_stopping = EarlyStopping('valAccTarget', mode='max', patience=10, verbose=True)
		trainer = Trainer(devices=1,
		                  accelerator="gpu",
		                  check_val_every_n_epoch=1,
		                  max_epochs=teacherParams['epoch'],
		                  logger=my_logger,
		                  enable_progress_bar=False,
		                  min_epochs=1,
		                  callbacks=[early_stopping],
		                  enable_model_summary=True,
		                  multiple_trainloader_mode='max_size_cycle')
		
		trainer.fit(model)
		predT = model.getPredict(domain='Target')
		predS = model.getPredict(domain='Source')
		accS = accuracy_score(predS['trueSource'], predS['predSource'])
		accT = accuracy_score(predT['trueTarget'], predT['predTarget'])
		f1S = f1_score(predS['trueSource'], predS['predSource'], average='weighted')
		f1T = f1_score(predT['trueTarget'], predT['predTarget'], average='weighted')
		cmS = confusion_matrix(predS['trueSource'], predS['predSource'])
		cmT = confusion_matrix(predT['trueTarget'], predT['predTarget'])
		# print('Source: ', accS, '  Target: ', accT)
		final_result["Acc Target"].append(accT)
		final_result["Acc Source"].append(accS)
		final_result["F1 Target"].append(f1T)
		final_result["F1 Source"].append(f1S)
		print(f'\n Acc target in trial {i}: {accT}\n')
		for class_ in range(4):
			final_result[f"Acc Target class {class_}"].append(cmT[class_][class_] / cmT[class_][:].sum())
			final_result[f"Acc Source class {class_}"].append(cmS[class_][class_] / cmS[class_][:].sum())
		
		if save_path is not None and accT > best_acc :
			print(f"saving: {dm_source.datasetName} to {dm_target.datasetName} with Acc {accT}\n\n")
			print(teacherParams)
			model.save_params(save_path, f'TeacherV4_{dm_source.datasetName}_{dm_target.datasetName}')
			best_acc = accT
		del model, trainer
	final_result['Target acc mean'] = MCI(final_result["Acc Target"])
	final_result['Source acc mean'] = MCI(final_result["Acc Source"])
	final_result['Target f1 mean'] = MCI(final_result["F1 Target"])
	final_result['Source f1 mean'] = MCI(final_result["F1 Source"])
	for class_ in range(4):
		final_result[f"Acc Target class {class_}"] = MCI(final_result[f"Acc Target class {class_}"])
		final_result[f"Acc Source class {class_}"] = MCI(final_result[f"Acc Source class {class_}"])
	return final_result


if __name__ == '__main__':
	print(f"params loaded from: {paramsPath}")
	teacherParams = getTeacherParams(paramsPath)
	teacherParams['input_shape'] = (2, args.freq * 2, 3)
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                freq=args.freq,
	                                input_shape=(2, args.freq * 2, 3),
	                                batch_size=128,
	                                oneHotLabel=True,
	                                shuffle=True)
	dm_source.setup(normalize=True)
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                input_shape=(2, args.freq * 2, 3),
	                                freq=args.freq,
	                                n_classes=args.n_classes,
	                                batch_size=128,
	                                oneHotLabel=True,
	                                shuffle=True)
	dm_target.setup(normalize=True)
	
	final_result = runDisc(teacherParams, dm_source, dm_target, args.trials, args.savePath, useMixup=True)
	print(final_result)
	if my_logger:
		my_logger.log_metrics(final_result)
