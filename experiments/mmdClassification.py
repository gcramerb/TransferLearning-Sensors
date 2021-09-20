import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim



import sys, os,argparse
import numpy as np


from sklearn.metrics import accuracy_score, recall_score, f1_score
# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.classifier import classifier
from models.customLosses import MMDLoss

from dataProcessing.create_dataset import crossDataset,targetDataset, getData
from Utils.trainer import Trainer



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


if __name__ == '__main__':
	
	hyp = dict()
	hyp['model'] = args.model
	hyp['loss'] = 'cel'
	hyp['penalty'] = args.penalty
	hyp['lr'] = 1e-3
	network = Trainer(hyp)
	network.configTrain(bs=256,n_ep=70)
	
	if args.inPath is None:
		args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	source = getData(args.inPath, args.source, getLabel = True)
	target = getData(args.inPath, args.target, getLabel = False)
	dataTrain = crossDataset(source, target)
	network.train(dataTrain)
	
	source = getData(args.inPath, args.source,getLabel= True)
	target = getData(args.inPath, args.target, getLabel=True)
	dataTest = crossDataset(source, target)
	yTrueSource,yPredSource, yTrueTarget,yPredTarget = network.predict(dataTest)
	
	print(hyp['model'],'  ',hyp['penalty'])
	print('Source: ')
	print('\n',accuracy_score(yTrueSource,yPredSource),'\n')
	print(recall_score(yTrueSource,yPredSource,average = 'macro'),'\n')
	print(f1_score(yTrueSource,yPredSource,average = 'macro'),'\n')

	print('Target: ')
	print('\n',accuracy_score(yTrueTarget,yPredTarget),'\n')
	print(recall_score(yTrueTarget,yPredTarget,average = 'macro'),'\n')
	print(f1_score(yTrueTarget,yPredTarget,average = 'macro'),'\n')

