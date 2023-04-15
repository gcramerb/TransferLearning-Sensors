import sys, argparse

sys.path.insert(0, '../')
import torch
from dataProcessing.dataModule import SingleDatasetModule
from Utils.params import getStudentParams
from Utils.train import getDatasets, runStudent

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--nClasses', type=int, default=4)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
	
_, dm_target = getDatasets(args.inPath, "Dsads", args.target, args.nClasses)
dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
                                     datasetName=f"",
                                     input_shape=2,
                                     n_classes=args.nClasses,
                                     batch_size=64,
                                     oneHotLabel=False,
                                     shuffle=True)

fileName = f"{args.source}_{args.target}pseudoLabel{args.model}.npz"
dm_pseudoLabel.setup(normalize=False, fileName=fileName)

def objective(trial):
	# Initialize the best_val_loss value
	clfParams= suggest_hyperparameters(trial)
	model = runStudent(studentParams, dm_pseudoLabel,dm_target)
	pred = model.predict(dm_target.test_dataloader())
	metrics = calculateMetrics(pred['pred'], pred['true'])
	acc = metrics["Acc"]
	if acc >= finalResult['top 1'][0]:
		finalResult['top 1'] = [result, clfParams]
		print(f'Result: {args.source} to {args.target}: \n --- {metrics}')
		print('clfParams: ', clfParams, '\n')
	return acc
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
if __name__ == '__main__':
	class_weight = None
	if args.nClasses ==4:
		class_weight = torch.tensor([0.5, 2, 2, 0.5])
	studentParams['class_weight'] = class_weight
	pre_train = True

	metrics = runStudent(studentParams, dm_pseudoLabel,dm_target)
	print(metrics['target'])
	if my_logger:
		my_logger.log_hyperparams(studentParams)
		my_logger.log_metrics(metrics['target'])
