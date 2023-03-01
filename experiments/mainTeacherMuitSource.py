import sys, argparse, os, glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule,MultiDatasetModule

import optuna
from trainers.trainerTL import TLmodel
from Utils.myUtils import MCI, getTeacherParams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=6)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--savePath', type=str, default=None)
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

#
# if args.log:
# 	my_logger = WandbLogger(project='Disc',
# 	                        log_model='all',
# 	                        name=args.expName + args.source + '_to_' + args.target)
finalResult = {}
finalResult['top 1'] = [0, {}]
datasetList = ["Uschad", "Dsads", "Ucihar"]
datasetList.remove(args.target)
if len(datasetList) != 2:
	raise ValueError('Dataset Name not exist')


dm_source =  MultiDatasetModule(data_dir=args.inPath,
			target= args.target,
			datasetList= datasetList,
			n_classes = args.n_classes,
			input_shape = 2 ,
			batch_size= 128,
			num_workers = 0,
			shuffle = False)
dm_source.setup(normalize=False)

dm_target = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                input_shape=2,
                                n_classes=args.n_classes,
                                batch_size=128,
                                oneHotLabel=False,
                                shuffle=True)
dm_target.setup(normalize=False, fileName=f"{args.target}_{args.n_classes}activities_to_{args.source}_{args.n_classes}activities.npz")


def runDisc(teacherParams, dm_source, dm_target):
	teacherParams['input_shape'] = dm_source.dataTrain.X.shape[1:]
	model = TLmodel(trainParams=teacherParams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                useMixup=False,
	                save_path=None,
	                class_weight=None)
	
	model.setDatasets(dm_source, dm_target)
	model.create_model()
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
	
	trainer.fit(model)
	
	predT = model.getPredict(domain='Target')
	predS = model.getPredict(domain='Source')
	accS = accuracy_score(predS['trueSource'], predS['predSource'])
	accT = accuracy_score(predT['trueTarget'], predT['predTarget'])
	return accS, accT


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
def objective(trial):
	save = False
	teacherParams = suggest_hyperparameters(trial)
	accS, accT = runDisc(teacherParams, dm_source, dm_target)
	
	if accT > finalResult['top 1'][0]:
		finalResult['top 1'] = [accT, teacherParams]
		print('\n------------------------------------------------\n')
		print(f'New Top 1: {accT} (Source: {accS})\n')
		print(teacherParams)
		print('\n------------------------------------------------\n\n\n')
	return acc
if __name__ == '__main__':
	run(args.trials)
	print(finalResult)

