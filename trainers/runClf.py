import sys
sys.path.insert(0, '../')

#import geomloss
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from trainers.trainerClf import ClfModel
from dataProcessing.dataModule import SingleDatasetModule

def runClassifier(dm,clfParams,my_logger = None):
	class_weight = None
	if dm.datasetName =='Pamap2':
		class_weight = torch.tensor([1.0, 1.5,1.8, 3.0])
	if dm.datasetName =='Uschad':
		class_weight = torch.tensor([0.5,3,3,0.5])

	model = ClfModel(lr=clfParams['lr'],
	                     n_classes=dm.n_classes,
	                     alpha=clfParams['alpha'],
	                     step_size=clfParams['step_size'],
	                     model_hyp=clfParams,
	                     weight_decay=clfParams['weight_decay'],
	                     class_weight=class_weight,
	                     input_shape=dm.input_shape)
	if my_logger:
		adicionalInfo = {}
		adicionalInfo['class_weight'] = class_weight
		my_logger.log_hyperparams(adicionalInfo)
		my_logger.watch(model)

	#early_stopping = EarlyStopping('val_loss', mode='min', min_delta=0.001, patience=10,verbose = True)

	trainer = Trainer(gpus=1,
	                  logger=my_logger,
	                  check_val_every_n_epoch=1,
	                  max_epochs=clfParams['epoch'],
	                  progress_bar_refresh_rate=1,
	                  callbacks=[])

	trainer.fit(model, datamodule=dm)
	metrics = model.get_all_metrics(dm)
	return trainer, model, metrics


