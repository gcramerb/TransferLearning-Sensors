import sys, argparse

sys.path.insert(0, '../')
import optuna
from Utils.params import getTeacherParams
from Utils.metrics import calculateMetricsFromTeacher
from Utils.train import getDatasets,calculateMetricsFromTeacher,runTeacher

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--nClasses', type=int, default=6)
parser.add_argument('--discrepancy', type=str, default="mmd")
parser.add_argument('--trials', type=int, default=1)
args = parser.parse_args()

my_logger = None

if args.slurm:
	verbose = 0
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'
dm_source,dm_target = getDatasets(args.inPath,args.source,args.target,args.nClasses)
def suggestTeacherHyp(trial):
	Tparams = getTeacherParams()
	Tparams['discrepancy'] = args.discrepancy
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
	k = int(trial.suggest_categorical("kernel2", [15,25]))
	Tparams['kernel_dim'] = [(5, 3), (k, 3)]
	return Tparams


finalResult = {}
finalResult['top 1'] = [0, {}]


def objective(trial):
	save = False
	teacherParams = suggestTeacherHyp(trial)
	model = runTeacher(teacherParams, dm_source, dm_target, args.nClasses)
	metrics = calculateMetricsFromTeacher(model)
	if metrics["Acc"] > finalResult['top 1'][0]:
		finalResult['top 1'] = [acc, teacherParams]
		print('\n------------------------------------------------\n')
		print(f'New Top 1: {acc}\n')
		print(teacherParams)
		print('\n------------------------------------------------\n\n\n')
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
