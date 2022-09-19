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
from Utils.myUtils import  MCI,getTeacherParams


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
parser.add_argument('--n_classes', type=int, default=4)
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
	args.savePath  = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\testsTeacher\\'

paramsPath = os.path.join(params_path, args.source[:3] + args.target[:3] + ".json")

def analizePL(teacherParams,selectionParams,source, target, savePseudoLabel = False):
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=(2,50,3),
	                                n_classes=args.n_classes,
	                                batch_size=128,
	                                shuffle=True)
	
	dm_target.setup(normalize=True)
	model = TLmodel(trainParams=teacherParams,
	                n_classes=args.n_classes,
	                lossParams=None,
	                save_path=None,
	                class_weight=None)
	
	model.setDatasets(dm_target = dm_target)
	model.create_model()
	model.load_params(args.savePath,f'Disc_class{source}_{target}')
	predT = model.getPredict(domain='Target')
	
	# from torchsummary import summary
	# summary(model.FE, (2, 50, 3))
	
	# early_stopping = EarlyStopping('loss', mode='min', patience=10, verbose=True)

	pred = {}
	pred['latent'] = predT['latentTarget']
	pred['pred']   = predT['predTarget']
	pred['true']   = predT['trueTarget']
	pred['probs']  = predT['probTarget']
	pred['data']   = predT['dataTarget']
	

	accIni = accuracy_score(pred['true'], pred['pred'])
	f1Ini = f1_score(pred['true'], pred['pred'],average = 'weighted')
	cm = confusion_matrix(pred['true'], pred['pred'])
	dataLen = len(pred['true'])
	print(f'INIT Acc: {accIni}\n F1Socre: {f1Ini}\n confusionMatrix: {cm}')
	print(f"INIT number of samples: {dataLen}")
	print("\n====================================================\n")
	
	best = 0
	for param_ in selectionParams['params']:
		result = 0
		print(f"\n\n METHOD: {selectionParams['method']}, param: {param_}\n")
		Xpl,softLabel, trueLabel = getPseudoLabel(pred.copy(),method = selectionParams['method'],param = param_)
		acc = accuracy_score(trueLabel,softLabel)
		cm = confusion_matrix(trueLabel, softLabel)
		f1 = f1_score(trueLabel, softLabel,average = 'weighted')
		print(f"number of samples: {len(trueLabel)}\n")
		print(f" %  of samples decrease: {100 - 100*len(trueLabel)/dataLen}\n")
		print(f'Acc: {acc}; Improovment: (+{(100*acc/accIni)-100}) \n F1 socre: {f1}; Improovment: (+{(100*f1/f1Ini) - 100}) \n confusionMatrix: {cm}\n=======================================================\n')
		
		for class_ in range(cm.shape[0]):
			result += cm[class_][class_] / cm[class_][:].sum()
		if result>best:
			best = result
			finalAcc = acc
			finalCM = cm
			finalF1 = f1
			finalNSamples = len(trueLabel)
			if savePseudoLabel:
				print(f'saving methdod ' ,selectionParams['method'], f'with param {param_}')
				fileName = f"{source}_{target}pseudoLabel.npz"
				path_file = os.path.join(args.inPath, fileName)
				with open(path_file, "wb") as f:
					np.savez(f, X=Xpl, y=softLabel, folds=np.zeros(1))
	print("\n========================================= BEST RESULT ==========================================\n")
	print(f"number of samples: {finalNSamples}\n")
	print(f" %  of samples decrease: {100 - 100 * finalNSamples/ dataLen}\n")
	print(
		f'Acc: {finalAcc}; Improovment: (+{(100 * finalAcc / accIni) - 100}) \n F1 socre: {finalF1}; Improovment: (+{(100 * finalF1 / f1Ini) - 100}) \n confusionMatrix: {finalCM}\n=======================================================\n')
	
	return True
if __name__ == '__main__':

	teacherParams = getTeacherParams(paramsPath)
	
	selectionParams = {}
	selectionParams['method'] = 'cluster'
	selectionParams['params'] = []
	for k in [128,64,32]:
		for l in [0.85,0.9]:
			for n in [50,100,150]:
				for n2 in [15, 30, 60,100]:
					param_i = {}
					param_i['nClusters'] = k
					param_i['labelConvergence'] = l
					param_i['minSamples'] = n
					param_i['minSamplesStep2'] = n2
					selectionParams['params'].append(param_i)
	analizePL(teacherParams,selectionParams,args.source, args.target, False)