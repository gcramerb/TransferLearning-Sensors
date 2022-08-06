import sys, argparse, os, glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from pytorch_lightning.loggers import WandbLogger
sys.path.insert(0, '../')
from pytorch_lightning import Trainer
from dataProcessing.dataModule import SingleDatasetModule
from models.pseudoLabSelection import getPseudoLabel
from trainers.trainerClf import ClfModel
from trainers.trainerTL import TLmodel
from Utils.myUtils import  MCI,getTeacherParams


"""
The main idea of this experiment is to get the pseudo label of the target by the trained
models and evaluate it by the hold labels
"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--expName', type=str, default='Stu_bench')
parser.add_argument('--TLParamsFile', type=str, default="DiscUscDsa.json")
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--savePath', type=str, default='../saved/teacherOficial/')
parser.add_argument('--source', type=str, default="Pamap2")
parser.add_argument('--target', type=str, default="Uschad")
parser.add_argument('--PLmethod', type=str, default="simplest")
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--saveModel', type=bool, default=True)
args = parser.parse_args()

my_logger = None
if args.slurm:
	verbose = 0
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/oficial/'
# my_logger = WandbLogger(project='TransferLearning-Soft-Label',
#                         log_model='all',
#                         name=args.expName + args.source + '_to_' + args.target)
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.savePath = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\Disc\\'


def analizePL(teacherParams,source, target):
	dm_target = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=target,
	                                input_shape=teacherParams['input_shape'],
	                                n_classes=args.n_classes,
	                                batch_size=teacherParams['bs'])
	
	dm_target.setup(normalize=True)
	dm_source = SingleDatasetModule(data_dir=args.inPath,
	                                datasetName=source,
	                                n_classes=args.n_classes,
	                                input_shape=teacherParams['input_shape'],
	                                batch_size=teacherParams['bs'])
	dm_source.setup(normalize=True)

	model = TLmodel(trainParams=teacherParams,
	                n_classes=4,
	                lossParams=None,
	                save_path=None,
	                class_weight=None)
	
	model.setDatasets(dm_source, dm_target)
	model.create_model()
	trainer = Trainer(gpus=1,
	                  check_val_every_n_epoch=1,
	                  max_epochs=teacherParams['epoch'],
	                  logger=None,
	                  min_epochs=1,
	                  progress_bar_refresh_rate=0,
	                  callbacks=[],
	                  enable_model_summary=False,
	                  multiple_trainloader_mode='max_size_cycle')
	trainer.fit(model)
	predT = model.getPredict(domain='Target')
	
	# from torchsummary import summary
	# summary(model.FE, (2, 50, 3))
	
	# early_stopping = EarlyStopping('loss', mode='min', patience=10, verbose=True)

	pred = {}
	pred['latent'] = predT['latentTarget']
	pred['pred'] = predT['predTarget']
	pred['true'] = predT['trueTarget']
	pred['probs'] = predT['probTarget']
	pred['data'] = predT['dataTarget']
	

	# teacher = ClfModel(lr=teacherParams['lr'],
	#                    n_classes=args.n_classes,
	#                    alpha=teacherParams['alpha'],
	#                    step_size=teacherParams['step_size'],
	#                    model_hyp=teacherParams,
	#                    weight_decay=teacherParams['weight_decay'],
	#                    class_weight=None,
	#                    input_shape=teacherParams['input_shape'])
	#
	# file = f'Disc_class{source}_{target}'
	# teacher.load_params(args.savePath, file)
	# pred = teacher.predict(dm_target.test_dataloader())
	
	
	acc = accuracy_score(pred['true'], pred['pred'])
	cm = confusion_matrix(pred['true'], pred['pred'])
	print(f'Acc: {acc}\n confusionMatrix: {cm}')
	print(f'INIT acc in Target{acc}')
	print(f"INIT number of samples: {len(pred['true'])}\n====================================================\n")
	
	methodParams = {}
	methodParams['cluster'] = [32,64,128,256]
	methodParams['simplest'] = [0.85,0.90,0.95,0.97]
	methodParams['kernel'] = [999] #dumb number
	
	for method in ['cluster','kernel','simplest']:
		for param_ in methodParams[method]:
			print(f"\n\n METHOD: {method}, param: {param_}\n")
			_,softLabel, trueLabel = getPseudoLabel(pred.copy(),method = method,param = param_)
			print(f"number of samples: {len(trueLabel)}\n")
			acc = accuracy_score(trueLabel,softLabel)
			cm = confusion_matrix(trueLabel, softLabel)
			print(f'Acc: {acc}\n confusionMatrix: {cm}\n=======================================================\n')
			del softLabel, trueLabel
	del teacher, dm_target
	return True
if __name__ == '__main__':
	path_TL_params = None
	if  args.TLParamsFile:
		path_TL_params = os.path.join(params_path, args.TLParamsFile)
	TLparams = getTeacherParams(path_TL_params)
	analizePL(TLparams,args.source, args.target)