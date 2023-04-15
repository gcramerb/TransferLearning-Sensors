from Utils.metrics import calculateMetrics,calculateMetricsFromTeacher
from dataProcessing.dataModule import SingleDatasetModule
from trainers.trainerTL import TLmodel
from trainers.trainerClf import ClfModel
from pytorch_lightning import Trainer
from Utils.params import getTeacherParams
import torch
def getDatasets(inPath,source,target,nClasses):
	dm_source = SingleDatasetModule(data_dir=inPath,
	                                datasetName="",
	                                n_classes=nClasses,
	                                input_shape=2,
	                                batch_size=128,
	                                oneHotLabel=False,
	                                shuffle=True)
	
	dm_source.setup(normalize=False,
	                fileName=f"{source}_to_{target}_{nClasses}activities.npz")
	dm_target = SingleDatasetModule(data_dir=inPath,
	                                datasetName="",
	                                input_shape=2,
	                                n_classes=nClasses,
	                                batch_size=128,
	                                oneHotLabel=False,
	                                shuffle=True)

	dm_target.setup(normalize=False,
	                fileName=f"{target}_to_{source}_{nClasses}activities.npz")
	
	return dm_source, dm_target
def suggestTeacherHyp(trial):
	Tparams = getTeacherParams()
	Tparams["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.7, step=0.1)
	Tparams['enc_dim'] = trial.suggest_categorical("enc_dim", [32,64, 128,256])
	Tparams['lr'] = 0.001
	Tparams['epoch'] = trial.suggest_int("epoch", 10, 80, step=10)
	Tparams['alpha'] = trial.suggest_float("alpha", 0.01, 3.0, step=0.05)
	Tparams['beta'] = trial.suggest_float("beta", 0.005, 0.5, step=0.0005)
	Tparams['weight_decay'] = trial.suggest_float("weight_decay", 0.0, 0.7, step=0.1)
	f1 = trial.suggest_int("f1", 2, 12, step=2)
	f2 = trial.suggest_int("f2", 12, 24, step=2)
	f3 = trial.suggest_int("f3", 24, 36, step=2)
	Tparams['n_filters'] = (f1, f2, f3)
	k = int(trial.suggest_categorical("kernel2", [15,25]))
	Tparams['kernel_dim'] = [(5, 3), (k, 3)]
	return Tparams
def runTeacher(teacherParams, dm_source, dm_target, classes=4):
	teacherParams['input_shape'] = dm_source.dataTrain.X.shape[1:]
	class_weight = None
	if classes == 4:
		class_weight = torch.tensor([0.5, 2, 2, 0.5])
	model = TLmodel(trainParams=teacherParams,
	                lossParams=None,
	                useMixup=False,
	                save_path=None,
	                class_weight=class_weight,
	                n_classes=classes)
	
	model.setDatasets(dm_source, dm_target)
	model.create_model()
	trainer = Trainer(devices=1,
	                  accelerator="gpu",
	                  check_val_every_n_epoch=1,
	                  max_epochs=teacherParams['epoch'],
	                  logger=None,
	                  enable_progress_bar=False,
	                  min_epochs=1,
	                  callbacks=[],
	                  enable_model_summary=True,
	                  multiple_trainloader_mode='max_size_cycle')
	trainer.fit(model)
	return model


def runTeacherNtrials(teacherParams, dm_source, dm_target, trials, save_path=None, classes=4):
	bestAcc = 0
	dictMetricsAll = []
	for i in range(trials):
		model = runTeacher(teacherParams,dm_source,dm_target,classes)
		metrics = calculateMetricsFromTeacher(model)
		dictMetricsAll.append(metrics)
		if metrics["Acc"] > bestAcc:
			bestAcc = accTarget
			if save_path is not None:
				print(f"saving: {dm_source.datasetName} to {dm_target.datasetName} with Acc {accT}\n\n")
				print(teacherParams)
				disc = teacherParams['dicrepancy']
				model.save_params(save_path, f'Teacher{disc}_{args.source}_{args.target}_{args.nClasses}actv')
	print(f'\n-------------------------------------------------------\n BEST Acc target {bestAcc}\n')
	print('-----------------------------------------------------------')
	return bestAcc, dictMetricsAll


def runStudent(studentParams, dm_pseudoLabel,dm_target):
	batchSize = 64
	studentParams['input_shape'] = dm_target.dataTrain.X.shape[1:]
	model = ClfModel(trainParams=studentParams,
	                 class_weight=studentParams['class_weight'],
	                 oneHotLabel=False,
	                 mixup=False)
	model.create_model()
	trainer = Trainer(devices=1,
	                  accelerator="gpu",
	                  check_val_every_n_epoch=1,
	                  max_epochs=studentParams["epochs"],
	                  enable_progress_bar=False,
	                  min_epochs=1,
	                  enable_model_summary=True)

	model.setDatasets(dm=dm_pseudoLabel, secondDataModule=dm_target)
	trainer.fit(model)
	return model
