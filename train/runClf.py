import sys
sys.path.insert(0, '../')

#import geomloss
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from train.trainerClf_pl import networkLight
from dataProcessing.dataModule import SingleDatasetModule

# def getModelHparams():
# 	clfParam = {}
# 	clfParam['kernel_dim'] = [(5, 3), (25, 1)]
# 	clfParam['n_filters'] = (6,8,24,24)
# 	clfParam['encDim'] =48
# 	clfParam["inputShape"] = (1,50,6)
#
# 	return clfParam
# def getTrainHparms():
# 	trainParams = {}
# 	trainParams['nEpoch'] = 30
# 	trainParams['batch_size'] = 64
# 	trainParams['alpha'] = 0.55
# 	trainParams['lr'] = 0.00015
# 	return trainParams
#
# def getHparamsPamap2():
# 	clfParam = {}
# 	clfParam['kernel_dim'] = [(5, 3), (25, 3)]
# 	clfParam['n_filters'] = (6,14,22,28)
# 	clfParam['encDim'] = 128
# 	clfParam["inputShape"] = (2,50,3)
# 	clfParam['DropoutRate'] = 0.2
# 	clfParam['FeName'] = 'fe1'
# 	trainParams = {}
# 	trainParams['nEpoch'] = 80
# 	trainParams['batch_size'] = 64
# 	trainParams['alpha'] = 0.0
# 	trainParams['lr'] = 0.00439
# 	trainParams['step_size'] = 15
# 	return clfParam,trainParams

def getModelHparams():
	clfParam = {}
	clfParam['kernel_dim'] = [(5, 3), (25, 3)]
	clfParam['n_filters'] = (4, 16, 18, 24)
	#clfParam['n_filters'] = (6, 16, 24, 32)
	clfParam['encDim'] = 64
	clfParam["inputShape"] = (2, 50, 3)
	clfParam['DropoutRate'] = 0.0
	clfParam['FE'] = 'fe2'
	
	return clfParam

def getTrainHparms():
	trainParams = {}
	trainParams['nEpoch'] = 100
	trainParams['batch_size'] = 64
	trainParams['alpha'] = .3
	trainParams['lr'] = 0.003
	trainParams['step_size'] = 50
	trainParams['weight_decay'] = 0
	
	return trainParams

def runClassifier(dm,my_logger = None, file_params = None,hparams = None):

	if file_params:
		pass
	if hparams:
		trainParams,clfParams = hparams
	else:
		clfParams = getModelHparams()
		trainParams = getTrainHparms()
	
	class_weight = None
	if dm.datasetName =='Pamap2':
		trainParams['nEpoch'] = 100
		class_weight = torch.tensor([1.0, 1.5,1.8, 3.0])
	if dm.datasetName =='Uschad':
		class_weight = torch.tensor([0.5,10,10,0.5])

	model = networkLight(alpha=trainParams['alpha'],
	                     lr=trainParams['lr'],
	                     n_classes = dm.n_classes,
	                     inputShape=clfParams["inputShape"],
	                     FeName=clfParams['FE'],
	                     step_size=trainParams['step_size'],
	                     weight_decay = trainParams['weight_decay'],
                         class_weight =class_weight,
	                     modelHyp=clfParams)
	if my_logger:
		my_logger.log_hyperparams(trainParams)
		my_logger.log_hyperparams(clfParams)
		my_logger.watch(model)
		
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=10,verbose = True)
	# chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True)
	# chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"

	trainer = Trainer(gpus=1,
	                  logger=my_logger,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['nEpoch'],
	                  progress_bar_refresh_rate=1,
	                  callbacks=[])

	trainer.fit(model, datamodule=dm)
	metrics = model.get_all_metrics(dm)

	return trainer, model, metrics

