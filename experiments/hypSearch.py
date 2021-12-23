import sys, argparse,time
#from geomloss import SamplesLoss
sys.path.insert(0,'../')

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from train.trainerClf_pl import networkLight
from train.runClf import runClassifier
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
	#setupTrain['alpha'] = trial.suggest_float("alpha", 0.0, 2.0, step=0.5)
	setupTrain['alpha'] = None
	setupTrain['bs'] = trial.suggest_categorical("bs",[64,128])
	setupTrain['step_size'] =  15
	setupTrain['nEpochs'] = 75

	modelHyp = {}
	modelHyp['kernel_dim'] = [(5, 3), (25, 3)]
	f1 = trial.suggest_int('filter 1',2,6,step=2)
	f2 = trial.suggest_int('filter 2',8,16,step=2)
	f3 = trial.suggest_int('filter 3',16,24,step=2)
	f4 = trial.suggest_int('filter 4',24,32,step=2)
	modelHyp['n_filters'] = (f1,f2,f3,f4)
	#modelHyp['n_filters'] = (4,16, 18, 24)
	modelHyp['encDim'] =  trial.suggest_int('encDim',64,129,step=8)
	modelHyp["DropoutRate"] = 0.2
	#FeName = trial.suggest_categorical('FeName', ['fe1', 'fe2'])
	modelHyp['FeName'] = 'fe2'
	modelHyp['inputShape'] =  (2,50,3)
	return setupTrain, modelHyp


def objective(trial):
	# Initialize the best_val_loss value
	best_loss_val = float("Inf")
	
	setupTrain, hypModel = suggest_hyperparameters(trial)
	dm = SingleDatasetModule(data_dir=args.inPath,
	                         datasetName=args.source,
	                         n_classes = args.n_classes,
	                         inputShape=hypModel['inputShape'],
	                         batch_size=setupTrain['bs'])
	dm.setup(Loso=False,split = True,normalize = True)
	
	hparams = (setupTrain, hypModel)
	trainer, model, res = runClassifier(dm,hparams = hparams)

	print(trial.number,' : ', res[0]['val_acc'])
	metric = res[0]['val_loss']
	if metric <= best_loss_val:
		best_loss_val = metric
	return best_loss_val

def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="minimize")
	study.optimize(objective, n_trials=n_trials)
	
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ',args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Loss (trial value): ", study.best_trial.value)
	
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':

	start = time.time()
	params = run(100)
	end = time.time()
	print((end - start)/60,'s for 100 trials')
