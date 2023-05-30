import sys, argparse,os
import numpy as np
import scipy.stats as st
import optuna
sys.path.insert(0, '../')
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from Utils.train import getDatasets
from Utils.metrics import calculateMetrics
from Utils.params import getTeacherParams
from trainers.trainerClf import ClfModel
from pytorch_lightning.callbacks import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--nClasses', type=int, default=6)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Uschad")
parser.add_argument('--trials', type=int, default=1)
args = parser.parse_args()
finalResult = {}
finalResult['top 1'] = [0, {}]
if args.slurm:
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'
else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\ot\\'
	args.log = False
def suggestTeacherHyp(trial):
	paramsPath = os.path.join(params_path,
	                          args.source[:3] + args.target[:3] + f"_{args.nClasses}activities_ot.json")
	studentParams = getTeacherParams(paramsPath)
	studentParams['batch_size'] = trial.suggest_categorical("batch_size", [32,64, 128,256])
	studentParams['lr'] =trial.suggest_loguniform("lr",0.0001,0.1)
	studentParams['epoch'] = trial.suggest_int("epoch", 5, 100, step=3)
	studentParams['centerLoss'] = trial.suggest_categorical("centerLoss",[True,False])
	return studentParams
def objective(trial):
	studentParams = suggestTeacherHyp(trial)
	dm_source, dm_target = getDatasets(args.inPath, args.source, args.target, args.nClasses, batchSize=studentParams['batch_size'])
	studentParams['input_shape'] = dm_target.dataTrain.X.shape[1:]
	model = ClfModel(trainParams=studentParams,
	                 n_classes =args.nClasses,
	                 oneHotLabel=False,
	                 mixup=False)
	model.create_model(setCenterLoss = studentParams['centerLoss'])
	seed_everything(42, workers=True)
	trainer = Trainer(devices=1,
	                  accelerator="gpu",
	                  check_val_every_n_epoch=1,
	                  max_epochs=studentParams["epoch"],
	                  callbacks=[EarlyStopping(monitor='valAcc (ps)',mode = "max",patience = 5)],
	                  enable_progress_bar=False,
	                  min_epochs=1,
	                  deterministic=True,
	                  enable_model_summary=True)

	model.setDatasets(dm=dm_source)
	trainer.fit(model)
	pred = model.predict(dm_target.test_dataloader())
	metrics= {}
	metrics['target'] = calculateMetrics(pred['pred'], pred['true'])
	pred = model.predict(dm_source.test_dataloader())
	metrics['source'] = calculateMetrics(pred['pred'], pred['true'])

	if metrics['source']["Acc"] > finalResult['top 1'][0]:
		print(args.target, ":\n")
		print(metrics['target'])
		print(args.source, ":\n")
		print(metrics['source'])
		print("\n\n\n______________________________________\n")
		finalResult['top 1'] = [metrics['source']["Acc"], studentParams]
		print('\n------------------------------------------------\n')
	return metrics['source']["Acc"]
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




