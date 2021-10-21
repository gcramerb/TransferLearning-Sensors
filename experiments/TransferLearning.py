import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.utils.data import DataLoader, random_split

import sys, os,argparse,pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.classifier import classifier
from models.customLosses import MMDLoss
from dataProcessing.create_dataset import crossDataset,targetDataset, getData
from dataProcessing.dataModule import CrossDatasetModule
from Utils.trainer import myTrainer,networkLight


import mlflow

from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint



from collections import OrderedDict

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='trialDef')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
parser.add_argument('--model', type=str, default="clf")
parser.add_argument('--penalty', type=str, default="clDist")
parser.add_argument('--batchS', type=int, default=128)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--alpha', type=float, default=1.2)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--saveModel', type=bool, default=False)

args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	
else:
	args.nEpoch = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	args.outPath = '../results/'
if __name__ == '__main__':
	dm = CrossDatasetModule(data_dir = args.inPath )
	dm.setup()
	model = networkLight(penalty = args.penalty, alpha=args.alpha)
	mlf_logger = MLFlowLogger(experiment_name=args.expName,save_dir = '../results/mlflow/')
	mlf_logger.log_hyperparams(params = {'penalty':args.penalty,'alpha': args.alpha,
	                                     'lr':args.lr,'source':args.source })
	
	early_stopping = EarlyStopping('val_loss',mode = 'min',patience=15)
	chkp_callback = ModelCheckpoint(dirpath='../saved/',save_last = True)
	chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"
	trainer = Trainer(callbacks=[early_stopping],gpus=AVAIL_GPUS,logger=mlf_logger,check_val_every_n_epoch =1, max_epochs=args.nEpoch, progress_bar_refresh_rate=0)
	trainer.fit(model, datamodule = dm)
	if args.saveModel:
		trainer.save_checkpoint(f"../saved/model1{args.source}_to_{args.target}.ckpt")
	print(f"{args.source}_to_{args.target}\n")
	print(trainer.test(datamodule=dm))
	mlf_logger.finalize()


