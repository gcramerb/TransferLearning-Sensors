import sys, argparse

sys.path.insert(0, '../')
import optuna
from Utils.params import getTeacherParams
from Utils.metrics import calculateMetricsFromTeacher
from Utils.train import getDatasets,calculateMetricsFromTeacher,runTeacher,suggestTeacherHyp

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



finalResult = {}
finalResult['top 1'] = [0, {}]


def objective(trial):
	save = False
	teacherParams = suggestTeacherHyp(trial)
	teacherParams['discrepancy'] = args.discrepancy
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
