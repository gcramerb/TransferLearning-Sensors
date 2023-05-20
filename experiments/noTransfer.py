import sys, argparse,os
import numpy as np
import scipy.stats as st

sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from Utils.train import getDatasets, runStudent
from Utils.metrics import calculateMetrics
from Utils.params import getTeacherParams
parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--nClasses', type=int, default=4)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
args = parser.parse_args()

datasetList = ["Dsads","Ucihar","Uschad"]
datasetList.remove(args.source)
if len(datasetList) != 2:
	raise ValueError('Dataset Name not exist')
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

def create_result_dict():
	result = {}
	for dat in datasetList:
		result[dat] = []
	return result

if __name__ == '__main__':
	result = create_result_dict()


	for dataset_i in datasetList:
		paramsPath = os.path.join(params_path,
		                          args.source[:3] + dataset_i[
		                                            :3] + f"_{args.nClasses}activities_ot.json")
		studentParams = getTeacherParams(paramsPath)
		dm_source, dm_target = getDatasets(args.inPath, args.source, datasetList[0], args.nClasses)
		studentParams['epoch'] = 70
		model = runStudent(studentParams, dm_source, dm_target, args.nClasses)
		pred = model.predict(dm_target.test_dataloader())
		metrics= {}
		metrics['target'] = calculateMetrics(pred['pred'], pred['true'])
		pred = model.predict(dm_source.test_dataloader())
		metrics['source'] = calculateMetrics(pred['pred'], pred['true'])
		result[dataset_i]  = metrics
		print(dataset_i,":\n")
		print(metrics)
		print("\n\n\n______________________________________\n")
