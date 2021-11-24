import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.utils.data import DataLoader, random_split

seed = 14
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

import sys, os, argparse, pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
np.random.seed(seed)
# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.classifier import classifier,classifierTest
from models.customLosses import MMDLoss
from dataProcessing.create_dataset import crossDataset, targetDataset, getData
from dataProcessing.dataModule import SingleDatasetModule

from Utils.myTrainer import myTrainer

import mlflow

# from pytorch_lightning.loggers import MLFlowLogger
# from pytorch_lightning import LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
#from Utils.trainerPL import  networkLight


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="ClDist")
parser.add_argument('--bs_source', type=int, default=32)
parser.add_argument('--bs_target', type=int, default=128)
parser.add_argument('--nEpoch', type=int, default=200)
parser.add_argument('--alpha', type=float, default=0.85)
parser.add_argument('--lr_source', type=float, default=0.001)
parser.add_argument('--lr_target', type=float, default=0.01)
parser.add_argument('--saveModel', type=bool, default=False)

args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.nEpoch = 100
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'

def getTrainSetup():
	trainSetup = dict()
	trainSetup['alpha'] = args.alpha
	trainSetup['nEpochs'] = args.nEpoch
	trainSetup['step_size'] = 50
	trainSetup['penalty'] = args.penalty
	trainSetup['lr'] = args.lr_source
	dm_source = SingleDatasetModule(data_dir=args.inPath, datasetName=args.source, batch_size=args.bs_source)
	dm_source.setup(Loso=True)
	dm_target = SingleDatasetModule(data_dir=args.inPath, datasetName=args.target, batch_size=args.bs_target)
	dm_target.setup(Loso=True)
	return trainSetup, dm_source, dm_target

def run():

	trainer = myTrainer('clf1')
	trainSetup, dm_source, dm_target = getTrainSetup()
	trainer.setupTrain(trainSetup, dm_source, dm_target,,
	trainHist = trainer.train()
	outcomes=trainer.predict(stage = 'test',metrics = True)
	print(trainHist)
	print(outcomes)
	trainer.save(f'../saved/clf_{args.source}.pkl')

if __name__ == "__main__":
	run()