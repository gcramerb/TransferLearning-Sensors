import sys, argparse,os
import numpy as np
import scipy.stats as st

sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from Utils.train import getDatasets
from Utils.metrics import calculateMetrics
from Utils.params import getTeacherParams
from trainers.trainerClf import ClfModel
from pytorch_lightning.callbacks import EarlyStopping
parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--nClasses', type=int, default=6)
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Ucihar")
parser.add_argument('--target', type=str, default="Uschad")
args = parser.parse_args()

if args.slurm:
	args.inPath = f'/storage/datasets/sensors/frankDatasets_{args.nClasses}actv/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	params_path = f'/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'
else:
	verbose = 1
	args.inPath = f'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset_{args.nClasses}actv\\'
	params_path = f'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\oficial\\ot\\'
	args.log = False

if __name__ == '__main__':

	paramsPath = os.path.join(params_path,
	                          args.source[:3] + args.target[:3] + f"_{args.nClasses}activities_ot.json")
	studentParams = getTeacherParams(paramsPath)
	dm_source, dm_target = getDatasets(args.inPath, args.source, args.target, args.nClasses)
	studentParams['epoch'] = 70
	studentParams['lr'] = 0.001
	batchSize = 128
	studentParams['input_shape'] = dm_target.dataTrain.X.shape[1:]
	model = ClfModel(trainParams=studentParams,
	                 n_classes =args.nClasses,
	                 oneHotLabel=False,
	                 mixup=False)
	model.create_model()
	
	trainer = Trainer(devices=1,
	                  accelerator="gpu",
	                  check_val_every_n_epoch=1,
	                  max_epochs=studentParams["epoch"],
	                  callbacks=[EarlyStopping(monitor='train_loss')],
	                  enable_progress_bar=True,
	                  min_epochs=1,
	                  enable_model_summary=True)

	model.setDatasets(dm=dm_source)
	trainer.fit(model)
	pred = model.predict(dm_target.test_dataloader())
	metrics= {}
	metrics['target'] = calculateMetrics(pred['pred'], pred['true'])
	pred = model.predict(dm_source.test_dataloader())
	metrics['source'] = calculateMetrics(pred['pred'], pred['true'])
	print(args.target,":\n")
	print(metrics)
	print("\n\n\n______________________________________\n")
