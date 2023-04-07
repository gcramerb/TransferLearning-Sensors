import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from pytorch_lightning.loggers import WandbLogger
sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
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
parser.add_argument('--savePath', type=str, default='../saved/teacherOficial/')
parser.add_argument('--dicrepancy', type=str, default="ot")
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Dsads")
parser.add_argument('--nClasses', type=int, default=6)
args = parser.parse_args()

my_logger = None
if args.slurm:
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'
else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\'
	#args.savePath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\teacherOficial\\'
	args.log = False

paramsPath = os.path.join(params_path, args.source[:3] + args.target[:3] + f"_{args.nClasses}activities.json")

def analizePL(predictions,selectionParam):

	print(f"\n\n METHOD: {selectionParams['method']}, param: {selectionParam}\n")
	Xpl,softLabel, trueLabel = getPseudoLabel(pred.copy(),method = selectionParams['method'],param = selectionParam,n_classes = args.nClasses)
	if len(Xpl)>0:
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
	selectionParams = {}
	selectionParams['method'] = 'cluster'
	useMixup = False
	selectionParamList  = getPLS_params()
	_,dm_target  = getDatasets(args.inPath,args.source,args.target,args.nClasses)
	teacherParams['input_shape'] = dm_target.dataTrain.X.shape[1:]
	
	model = TLmodel(trainParams=teacherParams,
	                n_classes=args.nClasses,
	                lossParams=None,
	                save_path=None,
	                class_weight=None)
	model.setDatasets(dm_target=dm_target)
	model.create_model()
	
	model.load_params(args.savePath, f'Teacher{args.dicrepancy}_{args.source}_{args.target}_{args.nClasses}actv')
	predictions = model.getPredict(domain='Target')
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

	best = 0
	paramFinal,finalAcc,finalCM= {},0,0
	for param in selectionParamList:
		selectionParams['params'] = param
		result, acc, cm, data = analizePL(pred,selectionParams)
		if result > best:
			best = result
			finalAcc, finalCM = acc, cm
			Xfinal,slFinal,trueFinal= data
			finalNSamples = len(trueFinal)
			paramFinal = param
	if best>0:
		print(f'saving methdod ', selectionParams['method'], f'with param {paramFinal}')
		fileName = f"{args.source}_{args.target}pseudoLabel_{args.nClasses}actv'.npz"
		path_file = os.path.join(args.inPath, fileName)
		with open(path_file, "wb") as f:
			np.savez(f, X=Xfinal, y=slFinal, yTrue=trueFinal, folds=np.zeros(1))
		print("\n========================================= BEST RESULT ==========================================\n")
		print(f"number of samples: {finalNSamples}\n")
		print(f" %  of samples decrease: {100 - 100 * finalNSamples / dataLen}\n")
		print(
			f'Acc: {finalAcc}; Improovment: (+{(100 * finalAcc / accIni) - 100});\n confusionMatrix: {finalCM}\n=======================================================\n')
	else:
		print('deu errado')