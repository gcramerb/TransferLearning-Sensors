import sys, argparse, os, glob

sys.path.insert(0, '../')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule
from train.runClf import runClassifier
from train.trainer_FT import FTmodel
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--n_classes', type=int, default=4)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	save_path = '../saved/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'


def suggest_hyperparameters(trial):
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['epoch'] = trial.suggest_categorical("epoch",  [5,10,15])
	clfParams["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.1)
	clfParams['FE'] = 'fe2'
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = None
	clfParams['bs'] = 128
	clfParams['lr'] = 0.00005
	clfParams['weight_decay'] =  trial.suggest_float("weight_decay", 0.0, 0.3, step=0.1)

	TLparams = {}
	TLparams['lr'] = 0.005
	TLparams['gan'] = False
	TLparams['lr_gan'] = 0.0
	TLparams['bs'] = 128
	TLparams['step_size'] = None
	TLparams['epoch'] = 50
	TLparams['feat_eng'] = 'asym'
	TLparams['alpha'] = 0.0
	TLparams['beta'] = 0.0
	TLparams['discrepancy'] = trial.suggest_categorical("discrepancy", ['mmd', 'ot'])
	TLparams['weight_decay'] = 0.0
	return clfParams,TLparams
	


def objective(trial):
	# Initialize the best_val_loss value
	best_metric = float(-1)
	clfParams,TLparams = suggest_hyperparameters(trial)
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.source,
	                                n_classes=args.n_classes,
	                                input_shape=clfParams['input_shape'],
	                                batch_size=clfParams['bs'])
	dm_source.setup(Loso=False, split=False, normalize=True)
	
	file = f'model_{args.source}'

	trainer, clf, res = runClassifier(dm_source, clfParams)
	print('Source: ', res)
	clf.save_params(save_path, file)
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                input_shape=clfParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=TLparams['bs'],
	                                type='target')
	dm_target.setup(Loso=False, split=False, normalize=True)
	
	model = FTmodel(trainParams=TLparams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                save_path=None,
	                model_hyp=clfParams)
	model.load_params(save_path, file)
	model.setDatasets(dm_source, dm_target)
	early_stopping = EarlyStopping('discpy_loss', mode='min', patience=10, verbose=True)
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=TLparams['epoch'],
	                  logger=None,
	                  progress_bar_refresh_rate=verbose,
	                  callbacks=[early_stopping],
	                  multiple_trainloader_mode='max_size_cycle')
	
	trainer.fit(model)
	res = model.get_final_metrics()
	print(res['acc_target_all'])
	if res['acc_target_all'] >= best_metric:
		best_metric = res['acc_target_all']
	return best_metric


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Acc (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	params = run(200)
