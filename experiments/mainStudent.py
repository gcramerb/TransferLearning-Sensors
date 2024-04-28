import sys, argparse
import os
sys.path.insert(0, '../')
import torch
from dataProcessing.dataModule import SingleDatasetModule
from Utils.train import getDatasets, runStudent
from Utils.metrics  import calculateMetrics
from Utils.params import getTeacherParams

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--multSource', action='store_true')
parser.add_argument('--dicrepancy', type=str, default="")
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--ParamsFile', type=str, default = None)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--nClasses', type=int, default=4)
args = parser.parse_args()

my_logger = None
originalDataPath = ""
if args.slurm:
	verbose = 0
	args.inPath = f'/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/studentOficial/'
	originalDataPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
	originalDataPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'

def train(studentParams):
    dm_pseudoLabel = SingleDatasetModule(data_dir=args.inPath,
                                        datasetName=f"",
                                        input_shape=2,
                                        n_classes=args.nClasses,
                                        batch_size=64,
                                        oneHotLabel=False,
                                        shuffle=True)

    _, dm_target = getDatasets(originalDataPath, args.source, args.target, args.nClasses)
    #fileName = f"{args.source}_{args.target}pseudoLabel_{args.nClasses}actv_{args.dicrepancy}.npz"
    fileName = f"{args.source}_{args.target}pseudoLabelV5.npz"
    dm_pseudoLabel.setup(normalize=False, fileName=fileName)
    model = runStudent(studentParams, dm_pseudoLabel,dm_target,nClasses = args.nClasses)
    pred = model.predict(dm_target.test_dataloader())
    metrics = calculateMetrics(pred['pred'], pred['true'])
    return metrics


if __name__ == '__main__':
	path_clf_params =  None
	if args.ParamsFile:
		path_clf_params = os.path.join(params_path,args.ParamsFile)
	params = getTeacherParams(path_clf_params)
	print(params)
	print("\n\n\n")
	finalResult = train(params)
	print(finalResult)