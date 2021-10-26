import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim



import sys, os,argparse,pickle
import numpy as np


from sklearn.metrics import accuracy_score, recall_score, f1_score
# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.classifier import classifier
from models.customLosses import MMDLoss

from dataProcessing.create_dataset import crossDataset,targetDataset, getData
from Utils.trainer_PL import Trainer

import mlflow


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="mmd")
args = parser.parse_args()

if args.slurm:
	n_ep = 50
else:
	n_ep = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'


if __name__ == '__main__':
	
	hyp = dict()
	hyp['model'] = args.model
	hyp['loss'] = 'cel'
	hyp['penalty'] = args.penalty
	hyp['lr'] = 0.014188514716320757
	network = Trainer(hyp)
	network.configTrain(bs=128,n_ep=n_ep)


	source, sourceVal = getData(args.inPath, args.source, getLabel = True,valRate =0.1)
	
	#we pass the label only to calculate the validation metrics.
	target ,targetVal = getData(args.inPath, args.target, getLabel = True,valRate = 0.1)
	dataTrain = crossDataset(source, target)
	dataVal =  crossDataset(sourceVal, targetVal,targetLab = True)
	del source
	del target
	del sourceVal
	del targetVal

	hist = network.train(dataTrain,printGrad = False,dataVal= dataVal)
	saveModel = os.path.relpath(f'../saved/{network.name}.pkl')
	with open(saveModel, 'wb') as outp:
		pickle.dump(network, outp, pickle.HIGHEST_PROTOCOL)

	print('Train Loss: ', hist)

	source = getData(args.inPath, args.source,getLabel= True)
	target = getData(args.inPath, args.target, getLabel=True)
	dataTest = crossDataset(source, target,targetLab = True)
	yTrueTarget,yTrueSource, yPredTarget, yPredSource = network.predict(dataTest)
	
	print(hyp['model'],'  ',hyp['penalty'])
	print('Source: ')
	print('\n Acc:',accuracy_score(yTrueSource,yPredSource),'\n')
	#print(recall_score(yTrueSource,yPredSource,average = 'macro'),'\n')
	print("F1: ",f1_score(yTrueSource,yPredSource,average = 'macro'),'\n')

	print('Target: ')
	print('\n',accuracy_score(yTrueTarget,yPredTarget),'\n')
	#print(recall_score(yTrueTarget,yPredTarget,average = 'macro'),'\n')
	print("F1: ",f1_score(yTrueTarget,yPredTarget,average = 'macro'),'\n')

