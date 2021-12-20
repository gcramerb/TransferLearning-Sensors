import sys
sys.path.insert(0, '../')

#import geomloss

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
	clfParam['encDim'] = 64
	clfParam["inputShape"] = (2, 50, 3)
	clfParam['DropoutRate'] = 0.2
	return clfParam

def getTrainHparms():
	trainParams = {}
	trainParams['nEpoch'] = 20
	trainParams['batch_size'] = 128
	trainParams['alpha'] = 0.5
	trainParams['lr'] = 0.0005
	trainParams['step_size'] = 20
	
	return trainParams

def runClassifier(dm,my_logger = None, file_params = None):

	if file_params:
		pass
	else:
		clfParams = getModelHparams()
		trainParams = getTrainHparms()
		
	if dm.datasetName =='Pamap2':
		trainParams['nEpoch'] = 75


	model = networkLight(alpha=trainParams['alpha'],
	                     lr=trainParams['lr'],
	                     n_classes = dm.n_classes,
	                     inputShape=clfParams["inputShape"],
	                     FeName='fe2',
	                     step_size=trainParams['step_size'],
	                     modelHyp=clfParams)
	if my_logger:
		my_logger.log_hyperparams(trainParams)
		my_logger.log_hyperparams(clfParams)
		#wandb_logger = WandbLogger(project='classifier', log_model='all', name='best_until_now')
	
	early_stopping = EarlyStopping('val_loss', mode='min', patience=3,verbose = True)
	# chkp_callback = ModelCheckpoint(dirpath='../saved/', save_last=True)
	# chkp_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_loss:.2f}-{accSourceTest:.2f}-last"
	#
	trainer = Trainer(gpus=1,
	                  logger=my_logger,
	                  check_val_every_n_epoch=1,
	                  max_epochs=trainParams['nEpoch'],
	                  progress_bar_refresh_rate=1,
	                  callbacks=[early_stopping])
	#wandb_logger.watch(model)
	trainer.fit(model, datamodule=dm)
	res = trainer.validate(model, datamodule=dm)

	return trainer, model, res

