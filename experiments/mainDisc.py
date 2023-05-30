import sys, argparse, os

sys.path.insert(0, '../')
from pytorch_lightning.loggers import WandbLogger
from Utils.params import getTeacherParams
from Utils.train import getDatasets, calculateMetricsFromTeacher, runTeacher, runTeacherNtrials
from dataProcessing.dataModule import SingleDatasetModule,MultiDatasetModule

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--multSource', action='store_true')
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--dicrepancy', type=str, default="ot")
parser.add_argument('--nClasses', type=int, default=4)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--pathToSave', type=str, default=None)
parser.add_argument('--params', type=str, default=None)
args = parser.parse_args()

if args.slurm:
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'

else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
	args.log = False
if args.params is None:
	args.params = args.source[:3] + args.target[:3] + f"_{args.nClasses}activities_{args.dicrepancy}.json"
paramsPath = os.path.join(params_path,args.params)
if __name__ == '__main__':
	datasetList = ["Dsads", "Ucihar", "Uschad"]
	datasetList.remove(args.target)
	if len(datasetList) != 2:
		raise ValueError('Dataset Name not exist')
	print(f"params loaded from: {paramsPath}")
	teacherParams = getTeacherParams(paramsPath)
	teacherParams['discrepancy'] = ""
	teacherParams['discrepancy'] = args.dicrepancy
	if args.multSource:
		dm_source = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName="",
		                                n_classes=6,
		                                input_shape=2,
		                                batch_size=128,
		                                oneHotLabel=False,
		                                shuffle=True)
		
		dm_source.setup(normalize=False,
		                fileName=f"{datasetList[0]}_and_{datasetList[1]}.npz")
		dm_source.datasetName = f"{datasetList[0]}_and_{datasetList[1]}"
		dm_target = SingleDatasetModule(data_dir=args.inPath,
		                                datasetName="",
		                                input_shape=2,
		                                n_classes=6,
		                                batch_size=128,
		                                oneHotLabel=False,
		                                shuffle=True)
		
		dm_target.setup(normalize=False,
		                fileName=f"{args.target}_MultiSource.npz")
		dm_target.datasetName = f"{args.target}_MultiSource"
	else:
		dm_source, dm_target = getDatasets(args.inPath, args.source, args.target, args.nClasses)
	metrics = runTeacherNtrials(teacherParams, dm_source, dm_target, args.trials, args.pathToSave, args.nClasses)
	print(metrics)
