import sys, argparse
#from geomloss import SamplesLoss
sys.path.insert(0,'../')

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from train.trainerClf_pl import networkLight
from dataProcessing.dataModule import SingleDatasetModule
import optuna


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--source', type=str, default="Pamap2")
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'


def suggest_hyperparameters(trial):
	setupTrain = {}
	setupTrain['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
	setupTrain['alpha'] = trial.suggest_float("alpha", 0.0, 2.0, step=0.5)
	setupTrain['bs'] = trial.suggest_categorical("bs",[64,128])
	setupTrain['step_size'] =  15
	setupTrain['nEpochs'] = 100

	modelHyp = {}
	modelHyp['kernel_dim'] = [(5, 3), (25, 3)]
	# f1 = trial.suggest_int('filter 1',2,6,step=2)
	# f2 = trial.suggest_int('filter 2',8,16,step=2)
	# f3 = trial.suggest_int('filter 3',16,24,step=2)
	# f4 = trial.suggest_int('filter 4',24,32,step=2)
	modelHyp['n_filters'] = (4,16, 18, 24)
	modelHyp['encDim'] =  trial.suggest_int('encDim',64,129,step=8)
	modelHyp["DropoutRate"] = 0.2
	FeName = trial.suggest_categorical('FeName', ['fe1', 'fe2'])
	#FeName = 'fe2'
	inputShape =  (2,50,3)
	return setupTrain, modelHyp, FeName, inputShape


def objective(trial):
	# Initialize the best_val_loss value
	best_loss_val = float("Inf")
	
	setupTrain, hypModel, FeName, inputShape = suggest_hyperparameters(trial)
	dm = SingleDatasetModule(data_dir=args.inPath,
	                         datasetName=args.source,
	                         n_classes = args.n_classes,
	                         inputShape=inputShape,
	                         batch_size=setupTrain['bs'])
	dm.setup(Loso=False)
	model = networkLight(alpha=setupTrain['alpha'],
	                     lr=setupTrain['lr'],
	                     step_size = setupTrain['step_size'],
	                     FeName = FeName,
	                     n_classes = args.n_classes,
	                     modelHyp = hypModel,
	                     inputShape = inputShape)
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=5)

	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=setupTrain['nEpochs'],
	                  progress_bar_refresh_rate=0,
	                  callbacks = [early_stopping])
	
	trainer.fit(model, datamodule=dm)
	res = trainer.validate(model, datamodule=dm)
	print(res[0]['val_acc'])
	metric = res[0]['val_loss']
	if metric <= best_loss_val:
		best_loss_val = metric
	return best_loss_val

def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ',args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Loss (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))
	
	#setupTrain, hypModel, FeName, inputShape = suggest_hyperparameters(study.best_trial)
	
	# #train and val in all datasets to check the quality of the classifier
	# for s in ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']:
	#
	# 	dm_source = SingleDatasetModule(data_dir=args.inPath,
	# 	                                datasetName=s,
	# 	                                inputShape=inputShape,
	# 	                                batch_size=setupTrain['bs'])
	# 	#only make sense the loso split if run all the k hold cross validadtion
	# 	dm_source.setup(Loso=False)
	# 	model = networkLight(alpha=setupTrain['alpha'],
	# 	                     lr=setupTrain['lr'],
	# 	                     modelHyp=hypModel,
	# 	                     FeName = FeName,
	# 	                     inputShape=inputShape)
	#
	# 	early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
	# 	wandb_logger = WandbLogger(project='classifier', log_model='all', name='hyp_search')
	# 	trainer = Trainer(gpus=1,
	# 	                  logger = wandb_logger,
	# 	                  check_val_every_n_epoch=5,
	# 	                  max_epochs=setupTrain['nEpochs'],
	# 	                  progress_bar_refresh_rate=5,
	# 	                  callbacks=[early_stopping])
	# 	trainer.fit(model, datamodule=dm_source)
	# 	res = trainer.validate(model, datamodule=dm_source)
	# 	metric = res[0]['val_acc']
	# 	print(metric)
	#
if __name__ == '__main__':
	params = run(100)
