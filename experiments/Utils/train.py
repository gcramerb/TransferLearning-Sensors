from Utils.metrics import calculateMetrics,calculateMetricsFromTeacher


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

def runTeacher(teacherParams, dm_source, dm_target, save_path=None, classes=4):
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
	                  logger=my_logger,
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
		model = runTeacher(teacherParams,dm_source,dm_target,save_path,classes)
		metrics = calculateMetricsFromTeacher(model)
		dictMetricsAll.append(metrics)
		if metrics["Acc"] > bestAcc:
			bestAcc = accTarget
			if save_path is not None:
				print(f"saving: {dm_source.datasetName} to {dm_target.datasetName} with Acc {accT}\n\n")
				print(teacherParams)
				model.save_params(save_path, f'Teacher{args.model}_{args.source}_{args.target}')
	print(f'\n-------------------------------------------------------\n BEST Acc target {bestAcc}\n')
	print('-----------------------------------------------------------')
	return bestAcc, dictMetricsAll


def runStudent(studentParams, dm_target, dm_pseudoLabel):
	batchSize = 64
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
	pred = model.predict(dm_target.test_dataloader())
	final_result = calculateMetrics(pred['pred'],pred['true'])
	final_result["Acc Target"].append(accuracy_score()
	return final_result

def runStudentNtrials