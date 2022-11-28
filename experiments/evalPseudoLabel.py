import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from pytorch_lightning.loggers import WandbLogger
sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from trainers.trainerTL import TLmodel
from Utils.myUtils import  MCI,getTeacherParams,getPLS_params


"""
The main idea of this experiment is to get the pseudo label of the target by the trained
models and evaluate it by the hold labels
"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='Stu_bench')
parser.add_argument('--TLParamsFile', type=str, default="DiscUscDsa.json")
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--savePath', type=str, default='../saved/teacherOficial_v2/')
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Pamap2")
parser.add_argument('--model', type=str, default="V4")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--freq', type=int, default=50)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/{args.model}/'

else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\V4\\'
	args.savePath  = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\teacherOficialV4\\'

paramsPath = os.path.join(params_path, "Disc" + args.source[:3] + args.target[:3] + ".json")


def analizePL(predictions,selectionParam):

	pred = {}
	pred['latent'] = predictions['latentTarget']
	pred['pred']   = predictions['predTarget']
	pred['true']   = predictions['trueTarget']
	pred['probs']  = predictions['probTarget']
	pred['data']   = predictions['dataTarget']
	
	accIni = accuracy_score(pred['true'], pred['pred'])
	f1Ini = f1_score(pred['true'], pred['pred'],average = 'weighted')
	cm = confusion_matrix(pred['true'], pred['pred'])
	dataLen = len(pred['true'])
	print(f'INIT Acc: {accIni}\n F1Socre: {f1Ini}\n confusionMatrix: {cm}')
	print(f"INIT number of samples: {dataLen}")
	print("\n====================================================\n")
	print(f"\n\n METHOD: {selectionParams['method']}, param: {selectionParam}\n")
	Xpl,softLabel, trueLabel = getPseudoLabel(pred.copy(),method = selectionParams['method'],param = selectionParam)
	if len(Xpl)>0 and cm.shape[0]== 4:
		acc = accuracy_score(trueLabel,softLabel)
		cm = confusion_matrix(trueLabel, softLabel)
		result = f1_score(trueLabel, softLabel,average = 'weighted')
		for class_ in range(cm.shape[0]-1):
			if cm[class_][class_] ==0 or cm[class_][:].sum()==0:
				result = result*0.8
			else:
				result += cm[class_][class_] / cm[class_][:].sum()
	else:
		acc, cm, result = 0, 0,0
	print(f"number of samples: {len(trueLabel)}\n")
	print(f" %  of samples decrease: {100 - 100*len(trueLabel)/dataLen}\n")
	print(f'Acc: {acc}; Improovment: (+{(100*acc/accIni)-100}); \n confusionMatrix: {cm}\n=======================================================\n')
	return result, acc,cm, (Xpl,softLabel, trueLabel)


if __name__ == '__main__':

	print(f"params loaded from: {paramsPath}")
	teacherParams = getTeacherParams(paramsPath)
	teacherParams['input_shape'] = (2, args.freq * 2, 3)
	selectionParams = {}
	selectionParams['method'] = 'cluster'

	selectionParamList  = getPLS_params()
		#{'nClusters': 128, 'labelConvergence':0.6, 'minSamples': 10}

	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=args.target,
	                                input_shape=(2, args.freq * 2, 3),
	                                n_classes=args.n_classes,
	                                batch_size=128,
	                                freq=args.freq,
	                                oneHotLabel=True,
	                                shuffle=True)
	
	dm_target.setup(normalize=True)
	model = TLmodel(trainParams=teacherParams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                save_path=None,
	                class_weight=None)
	model.setDatasets(dm_target=dm_target)
	model.create_model()
	model.load_params(args.savePath, f'Teacher{args.model}_{args.source}_{args.target}')
	predictions = model.getPredict(domain='Target')
	accIni = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
	dataLen = len(predictions['trueTarget'])
	Predictionsfinal = []
	best = 0
	for param in selectionParamList:
		selectionParams['params'] = param
		result, acc, cm, data = analizePL(predictions,selectionParams)
		if result > best:
			best = result
			finalAcc, finalCM = acc, cm
			Xfinal,slFinal,trueFinal= data
			finalNSamples = len(trueFinal)
			paramFinal = param
	print(f'saving methdod ', selectionParams['method'], f'with param {paramFinal}')
	fileName = f"{args.source}_{args.target}pseudoLabel{args.model}.npz"
	path_file = os.path.join(args.inPath, fileName)
	with open(path_file, "wb") as f:
		np.savez(f, X=Xfinal, y=slFinal, yTrue=trueFinal, folds=np.zeros(1))
	print("\n========================================= BEST RESULT ==========================================\n")
	print(f"number of samples: {finalNSamples}\n")
	print(f" %  of samples decrease: {100 - 100 * finalNSamples / dataLen}\n")
	print(
		f'Acc: {finalAcc}; Improovment: (+{(100 * finalAcc / accIni) - 100});\n confusionMatrix: {finalCM}\n=======================================================\n')
