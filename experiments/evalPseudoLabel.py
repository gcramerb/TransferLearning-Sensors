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
	result = 0
	for param_i in selectionParams['params']:
		print(f"\n\n METHOD: {selectionParams['method']}, param: {param_i}\n")
		Xpl,softLabel, trueLabel = getPseudoLabel(pred.copy(),method = selectionParams['method'],param = param_i)
		if len(Xpl)>0:
			acc = accuracy_score(trueLabel,softLabel)
			cm = confusion_matrix(trueLabel, softLabel)
			f1 = f1_score(trueLabel, softLabel,average = 'weighted')
			for class_ in range(cm.shape[0]):
				if cm[class_][:].sum() > 0 and class_ != 3:
					result += cm[class_][class_] / cm[class_][:].sum()
		else:
			acc, cm, f1 = 0, 0,0
		print(f"number of samples: {len(trueLabel)}\n")
		print(f" %  of samples decrease: {100 - 100*len(trueLabel)/dataLen}\n")
		print(f'Acc: {acc}; Improovment: (+{(100*acc/accIni)-100}) \n F1 socre: {f1}; Improovment: (+{(100*f1/f1Ini) - 100}) \n confusionMatrix: {cm}\n=======================================================\n')
		
		if result>best:
			best = result
			finalAcc, finalCM,finalF1  = acc,cm,f1
			finalNSamples = len(trueLabel)
			Xfinal, slFinal, trueFinal=  Xpl,softLabel, trueLabel
			paramFinal = param_i

	print("\n========================================= BEST RESULT ==========================================\n")
	print(f"number of samples: {finalNSamples}\n")
	print(f" %  of samples decrease: {100 - 100 * finalNSamples/ dataLen}\n")
	print(
		f'Acc: {finalAcc}; Improovment: (+{(100 * finalAcc / accIni) - 100}) \n F1 socre: {finalF1}; Improovment: (+{(100 * finalF1 / f1Ini) - 100}) \n confusionMatrix: {finalCM}\n=======================================================\n')
	if savePseudoLabel:
		print(f'saving methdod ', selectionParams['method'], f'with param {paramFinal}')
		fileName = f"{source}_{target}pseudoLabel2steps.npz"
		path_file = os.path.join(args.inPath, fileName)
		with open(path_file, "wb") as f:
			np.savez(f, X=Xfinal, y=slFinal, yTrue=trueFinal, folds=np.zeros(1))
	
	return True
if __name__ == '__main__':

	teacherParams = getTeacherParams(paramsPath)
	
	selectionParams = {}
	selectionParams['method'] = 'cluster'
	selectionParams['params'] = []
	for k in [8,64,128,256,512]:
		for l in [0.5,0.6]:
				for n in [10, 30]:
					param_i = {}
					param_i['nClusters'] = k
					param_i['labelConvergence'] = l
					param_i['minSamples'] = n
					selectionParams['params'].append(param_i)
	# selectionParams['params'] ={'nClusters':64 , 'labelConvergence':0.9 , 'minSamples':100 , 'minSamplesStep2':60}
	# bestParms = {}
	# bestParms['Ucihar_Pamap2'] = selectionParams
	# selectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 50, 'minSamplesStep2': 60}
	# bestParms['Ucihar_Dsads'] =selectionParams
	# selectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.6, 'minSamples': 100, 'minSamplesStep2': 60}
	# bestParms['Dsads_Pamap2'] =selectionParams
	# selectionParams['params'] = {'nClusters': 32, 'labelConvergence':0.85, 'minSamples': 150, 'minSamplesStep2': 30}
	# bestParms['Dsads_Uschad'] =selectionParams
	# selectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.6, 'minSamples': 100, 'minSamplesStep2': 30}
	# bestParms['Uschad_Dsads'] =selectionParams
	# selectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 50, 'minSamplesStep2': 15}
	# bestParms['Uschad_Ucihar'] =selectionParams
	# selectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 50, 'minSamplesStep2': 60}
	# bestParms['Pamap2_Dsads'] =selectionParams
	# selectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 100, 'minSamplesStep2': 15}
	# bestParms['Pamap2_Ucihar'] =selectionParams
	analizePL(teacherParams,selectionParams,args.source, args.target, True)