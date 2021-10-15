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

from models.classifier import classifier, networkLight
from models.customLosses import MMDLoss

from dataProcessing.create_dataset import crossDataset,targetDataset, getData
from dataProcessing.dataModule import CrossDatasetModule
from Utils.trainer import Trainer

import mlflow

from collections import OrderedDict

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)

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
	dm = CrossDatasetModule(data_dir = args.inPath )
	model = networkLight()
	trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=n_ep, progress_bar_refresh_rate=20)
	trainer.fit(model, dm)