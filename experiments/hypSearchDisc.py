import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from trainers.trainerTL import TLmodel
import optuna
from Utils.myUtils import MCI, getTeacherParams

import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=6)
parser.add_argument('--trasholdToSave', type=float, default=0)
parser.add_argument('--trials', type=int, default=1)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.n_classes}actv/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	save_path = '../saved/hypDisc/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/V5/'

else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.n_classes}actv\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'

def suggest_hyperparameters(trial):

	Tparams = getTeacherParams()
	Tparams["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.7, step=0.1)
	Tparams['enc_dim'] = trial.suggest_categorical("enc_dim", [32,64, 128,256])
	Tparams['lr'] = 0.001
	Tparams['epoch'] = trial.suggest_int("epoch", 10, 80, step=10)
	Tparams['alpha'] = trial.suggest_float("alpha", 0.01, 3.0, step=0.05)
	Tparams['beta'] = trial.suggest_float("beta", 0.005, 0.5, step=0.0005)
	Tparams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	f1 = trial.suggest_int("f1", 2, 12, step=2)
	f2 = trial.suggest_int("f2", 12, 24, step=2)
	f3 = trial.suggest_int("f3", 24, 36, step=2)
	Tparams['n_filters'] = (f1, f2, f3)
	return Tparams


finalResult = {}
finalResult['top 1'] = [0, {}]
finalResult['top 2'] = [0, {}]
finalResult['top 3'] = [0, {}]
finalResult['count'] = 0
dm_source = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                n_classes=args.n_classes,
                                input_shape=2,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)

dm_source.setup(normalize=False, fileName=f"{args.source}_{args.n_classes}activities_to_{args.target}_{args.n_classes}activities.npz")
dm_target = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                input_shape=2,
                                n_classes=args.n_classes,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)
dm_target.setup(normalize=False, fileName=f"{args.target}_{args.n_classes}activities_to_{args.source}_{args.n_classes}activities.npz")


def calculateMetrics(pred, true):
	final_result = {}
	final_result["Acc"] = []
	final_result["F1"] = []
	for class_ in range(4):
		final_result[f"Acc class {class_}"] = []
	acc = accuracy_score(pred, true)
	return acc, final_result

def runDisc(teacherParams, dm_source, dm_target, trials, save_path=None, classes=4):
	best_acc = 0
	dictMetricsAll = []
	for i in range(trials):
		teacherParams['input_shape'] = dm_source.dataTrain.X.shape[1:]
		class_weight = None
		if classes == 4:
			class_weight = torch.tensor([0.5, 2, 2, 0.5])
		model = TLmodel(trainParams=teacherParams,
		                lossParams=None,
		                useMixup=False,
		                save_path=None,
		                class_weight=class_weight,
		                n_classes=classes)
		
		model.setDatasets(dm_source, dm_target)
		model.create_model()
		#from torchsummary import summary
		# summary(model.to("cuda").FE, teacherParams['input_shape'])
		# model.load_params(args.savedPath, f'Teacher{args.model}_{args.source}_{args.target}')
		
		# early_stopping = EarlyStopping('valAccTarget', mode='max', patience=10, verbose=True)
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
		accT, dictMetricsT = calculateMetrics(predT['trueTarget'], predT['predTarget'])
		dictMetricsAll.append(dictMetricsT)
		if accT > best_acc:
			best_acc = accT
			if save_path is not None:
				print(f"saving: {dm_source.datasetName} to {dm_target.datasetName} with Acc {accT}\n\n")
				print(teacherParams)
				model.save_params(save_path, f'Teacher{args.model}_{args.source}_{args.target}')
		
		del model, trainer
	print(f'\n-------------------------------------------------------\n BEST Acc target {best_acc}\n')
	print('-----------------------------------------------------------')
	return best_acc, dictMetricsAll
def objective(trial):
	save = False
	teacherParams = suggest_hyperparameters(trial)
	acc, dictMetricsAll = runDisc(teacherParams, dm_source, dm_target, 1,None, args.n_classes)
	
	if acc > finalResult['top 1'][0]:
		finalResult['top 1'] = [acc, teacherParams]
		print('\n------------------------------------------------\n')
		print(f'New Top 1: {acc}\n')
		print(teacherParams)
		print('\n------------------------------------------------\n\n\n')
	elif acc > finalResult['top 2'][0]:
		finalResult['top 2'] = [acc, teacherParams]
	elif acc > finalResult['top 3'][0]:
		finalResult['top 3'] = [acc, teacherParams]
	finalResult['count'] += 1
	return acc


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print('Target dataset: ', args.target)
	print("  Trial number: ", study.best_trial.number)
	print("  Acc (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))
	return
if __name__ == '__main__':
	run(args.trials)
	print(finalResult)
