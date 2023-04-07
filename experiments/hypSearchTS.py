import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, '../')

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from trainers.trainerClf import ClfModel
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Uschad")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--n_classes', type=int, default=0)
parser.add_argument('--trials', type=int, default=1)

args = parser.parse_args()
finalResult = {}
finalResult['top 1'] = [0, {}]
if args.slurm:
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.n_classes}actv/'
	verbose = 0
else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.n_classes}actv\\'


def suggest_hyperparameters(trial):
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	f1 = trial.suggest_int("f1", 4, 12, step=2)
	f2 = trial.suggest_int("f2", 12, 24, step=2)
	f3= trial.suggest_int("f3", 24, 32, step=2)
	clfParams['n_filters'] = (f1,f2,f3)
	clfParams['enc_dim'] = trial.suggest_categorical("enc_dim", [32,64, 128,256])
	clfParams['alpha'] = trial.suggest_float("alpha", 0.01, 3.0, step=0.05)
	clfParams['step_size'] = None
	clfParams['epoch'] = trial.suggest_int("epoch", 5, 86, step=10)
	clfParams["dropout_rate"] =  trial.suggest_float("dropout_rate", 0.0, 0.7, step=0.1)
	clfParams['bs'] = 128
	clfParams['lr'] =  trial.suggest_float("lr", 1e-5, 1e-3, log=True)
	clfParams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	return clfParams


dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
                                     datasetName=f"",
                                     input_shape=2,
                                     n_classes=args.n_classes,
                                     batch_size=64,
                                     oneHotLabel=False,
                                     shuffle=True)

dm_pseudoLabel.setup(normalize=False, fileName=f"{args.source}_{args.target}pseudoLabel_{args.n_classes}actv'.npz")

dm_target = SingleDatasetModule(data_dir=args.inPath,
                                datasetName="",
                                input_shape=2,
                                n_classes=args.n_classes,
                                batch_size=64,
                                oneHotLabel=False,
                                shuffle=True)
dm_target.setup(normalize=False, fileName=f"{args.target}_to_{args.source}_{args.n_classes}activities.npz")

def runStudent(studentParams,  class_weight=None):
	batchSize = 64
	studentParams['input_shape'] = dm_target.dataTrain.X.shape[1:]
	model = ClfModel(trainParams=studentParams,
	                 n_classes = args.n_classes,
	                 class_weight=class_weight)
	model.setDatasets(dm=dm_pseudoLabel, secondDataModule=dm_target)
	model.create_model()
	# early_stopping = EarlyStopping('training_loss', mode='min', patience=10, verbose=True)
	early_stopping = []
	log = int(dm_target.X_train.__len__() / batchSize)
	
	trainer = Trainer(devices=1,
	                  accelerator="gpu",
	                  check_val_every_n_epoch=1,
	                  max_epochs=studentParams['epoch'],
	                  logger=None,
	                  enable_progress_bar=False,
	                  min_epochs=1,
	                  log_every_n_steps=log,
	                  callbacks=early_stopping,
	                  enable_model_summary=True)
	trainer.fit(model)
	pred = model.predict(dm_target.test_dataloader())
	return accuracy_score(pred['true'], pred['pred'])

def objective(trial):
	# Initialize the best_val_loss value
	clfParams= suggest_hyperparameters(trial)
	result = runStudent(clfParams)
	print('Student acc Target: ',result)
	if result >= finalResult['top 1'][0]:
		finalResult['top 1'] = [result, clfParams]
		print(f'Result: {args.source} to {args.target} --- {result}')
		print('clfParams: ', clfParams, '\n')
	return result


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print('Target dataset: ', args.target)
	print(" Trial number: ", study.best_trial.number)
	print("  Acc (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	run(args.trials)
	print(finalResult)
