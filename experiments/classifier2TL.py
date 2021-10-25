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
from dataProcessing.dataModule import CrossDatasetModule
from Utils.trainer import  networkLight

import mlflow

from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from collections import OrderedDict

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = 1
BATCH_SIZE = 128


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="clDist")
parser.add_argument('--batchS', type=int, default=128)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--saveModel', type=bool, default=False)

args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'

else:
	args.nEpoch = 5
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/tests/'
if __name__ == '__main__':
	dm = CrossDatasetModule(data_dir=args.inPath)
	dm.setup()
	model = networkLight(penalty=args.penalty, alpha=args.alpha, lr=args.lr)
	mlf_logger = MLFlowLogger(experiment_name=args.expName, save_dir='../results/mlflow/')
	mlf_logger.log_hyperparams(params={'penalty': args.penalty, 'alpha': args.alpha,
	                                   'lr': args.lr, 'source': args.source})
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=40)
	chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True)
	chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"
	trainer = Trainer(gpus = 1,logger=mlf_logger, check_val_every_n_epoch=0,
	                  max_epochs=args.nEpoch, progress_bar_refresh_rate=0)
	trainer.fit(model, datamodule=dm)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/model1{args.source}_to_{args.target}.ckpt")
	print(f"{args.source}_to_{args.target}\n")
	print(trainer.test(datamodule=dm))
	mlf_logger.finalize()


