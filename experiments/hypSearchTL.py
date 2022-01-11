import sys, argparse, os, glob


sys.path.insert(0, '../')

from pytorch_lightning import Trainer
from train.trainer_FT import FTmodel
from train.runClf import runClassifier
from dataProcessing.dataModule import SingleDatasetModule
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--expName', type=str, default='hypSrch')
parser.add_argument('--save_path', type=str, default='../saved/')
parser.add_argument('--n_classes', type=int, default=4)

args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'
	verbose = 1
	args.save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\'


def suggest_hyperparameters(trial):
	params = {}
	params['lr_source'] =0.0001
	params['lr_target'] = trial.suggest_float("lr_target", 1e-6, 1e-1, log=True)
	params['lr_gan'] = trial.suggest_float("lr_gan", 1e-6, 1e-1, log=True)
	params['alphaS'] = None
	params['betaT'] = trial.suggest_float("alphaT", 0.1, 2.1, step=0.2)
	params['alphaT'] = trial.suggest_float("alphaT", 0.1, 2.1, step=0.2)
	
	bs = trial.suggest_categorical("batch_size", [128,256])
	params['bs_source'] = bs
	params['bs_target'] = bs
	params['step_size'] = 25
	params['n_epch'] = 1
	params['epch_rate'] =15
	params['discrepancy'] = trial.suggest_categorical("discrepancy", ['mmd','ot'])
	params['feat_eng'] =  trial.suggest_categorical("feat_eng", ['asym','sym'])
	params['weight_decay'] =0.3
	params['input_shape'] = (2, 50, 3)
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['encDim'] = 64
	clfParams["DropoutRate"] =0.2
	clfParams['FeName'] = 'fe2'
	
	# lossParams = {}
	# lossParams['blur'] = trial.suggest_float("blur", 0.005, 0.05, step=0.01)
	# lossParams['scaling'] = trial.suggest_float("scaling", 0.5, 0.9, step=0.2)
	# lossParams['debias'] = trial.suggest_categorical("debias", [True, False])
	lossParams = None
	return params, clfParams,lossParams



def objective(trial):
	# Initialize the best_val_loss value
	best_metric = float(-1)
	
	trainParams,modelParams,_ = suggest_hyperparameters(trial)

	
	dm_source = SingleDatasetModule(data_dir=args.inPath, datasetName=args.source, n_classes=args.n_classes,
	                                input_shape=trainParams['input_shape'], batch_size=trainParams['bs_source'])
	dm_source.setup(Loso=False, split=False,normalize = True)
	file = f'model_{args.source}'
	if os.path.join(args.save_path,file + '_feature_extractor') not in glob.glob(args.save_path + '*'):
		trainer, clf, res = runClassifier(dm_source)
		print('Source: ',res)
		clf.save_params(args.save_path,file)
	
	dm_target = SingleDatasetModule(data_dir=args.inPath, datasetName=args.target, n_classes=args.n_classes,
	                                input_shape=trainParams['input_shape'], batch_size=trainParams['bs_target'],
	                                type='target')
	dm_target.setup(Loso=False, split=False,normalize = True)

	model = FTmodel(lr=trainParams['lr_target'], lr_gan=trainParams['lr_gan'], n_classes=args.n_classes,
	                alpha=trainParams['alphaT'], beta=trainParams['betaT'], penalty=trainParams['discrepancy'],
	                input_shape=trainParams['input_shape'], model_hyp=modelParams,
	                weight_decay=trainParams['weight_decay'], feat_eng=trainParams['feat_eng'],
	                FE=modelParams['FeName'])
	
	model.load_params(args.save_path,file)
	model.setDatasets(dm_source, dm_target)
	
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['n_epch'],
	                  logger=None,
	                  progress_bar_refresh_rate=verbose,
	                  # callbacks = [early_stopping],
	                  multiple_trainloader_mode='max_size_cycle')
	
	trainer.fit(model)
	res = trainer.test(model=model)
	print(type(res))

	if res[0]['test_acc_target'] >= best_metric:
		best_metric = res[0]['test_acc_target']
	return best_metric


def run(n_trials):
	study = optuna.create_study(study_name="pytorch-mlflow-optuna", direction="maximize")
	study.optimize(objective, n_trials=n_trials)
	print("\n++++++++++++++++++++++++++++++++++\n")
	print('Source dataset: ', args.source)
	print("  Trial number: ", study.best_trial.number)
	print("  Loss (trial value): ", study.best_trial.value)
	print("  Params: ")
	for key, value in study.best_trial.params.items():
		print("    {}: {}".format(key, value))

if __name__ == '__main__':
	params = run(100)
