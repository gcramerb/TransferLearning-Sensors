import sys, argparse, os, glob
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, '../')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataProcessing.dataModule import SingleDatasetModule

from trainers.runClf import runClassifier
from trainers.trainerTL import TLmodel
from Utils.myUtils import get_Clfparams, get_TLparams, MCI

seed = 2804
print('Seeding with {}'.format(seed))
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--TLParamsFile', type=str, default=None)
parser.add_argument('--ClfParamsFile', type=str, default=None)
parser.add_argument('--n_classes', type=int, default=4)
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/frankDatasets/'
	args.outPath = '/mnt/users/guilherme.silva/TransferLearning-Sensors/results'
	verbose = 0
	save_path = '../saved/Disc/'
	params_path = '/mnt/users/guilherme.silva/TransferLearning-Sensors/experiments/params/'
else:
	verbose = 1
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	params_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\experiments\\params\\'
	args.paramsPath = None
	save_path = 'C:\\Users\\gcram\\Documents\\GitHub\\TransferLearning-Sensors\\saved\\Disc\\'





def runAnalise(clfParams, TLparams):
	
	final_result = {}
	datasets = ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']
	for source in datasets:
		for target in datasets:
			if source != target:
		
				dm_source = SingleDatasetModule(data_dir=args.inPath,
				                                datasetName=source,
				                                n_classes=args.n_classes,
				                                input_shape=clfParams['input_shape'],
				                                batch_size=TLparams['bs'])
				dm_source.setup(normalize=True)
				dm_target = SingleDatasetModule(data_dir=args.inPath,
				                                datasetName=target,
				                                input_shape=clfParams['input_shape'],
				                                n_classes=args.n_classes,
				                                batch_size=TLparams['bs'])
				dm_target.setup(normalize=True)
				
				model = TLmodel(trainParams=TLparams,
				                n_classes=args.n_classes,
				                lossParams=None,
				                save_path=None,
				                class_weight=None,
				                model_hyp=clfParams)
				
				
				model.setDatasets(dm_source, dm_target)
				model.create_model()
				file = f'Disc_{source}_{target}'
				model.load_params(save_path, file)
	
				predT = model.getPredict(domain='Target')
				predS = model.getPredict(domain='Source')
			
				accS = accuracy_score(predS['trueSource'], predS['predSource'])
				accT = accuracy_score(predT['trueTarget'], predT['predTarget'])
				#print('Source: ', accS, '  Target: ', accT,'\n\n')
				cmS = confusion_matrix(predS['trueSource'], predS['predSource'])
				cmT = confusion_matrix(predT['trueTarget'], predT['predTarget'])
				#print('Source: ', cmS, '  Target: ', cmT,'\n\n')
				final_result[f'{source} to {target} acc (S)'] = accS
				final_result[f'{source} to {target} acc (T)'] = accT
				final_result[f'{source} to {target} cm (S)'] = str(cmS)
				final_result[f'{source} to {target} cm (T)'] = str(cmT)
				del model, dm_target, dm_source
	return final_result


if __name__ == '__main__':
	path_clf_params, path_TL_params = None, None
	if args.ClfParamsFile:
		path_clf_params = os.path.join(params_path, args.ClfParamsFile)
	if args.TLParamsFile:
		path_TL_params = os.path.join(params_path, args.TLParamsFile)
	
	clfParams = get_Clfparams(path_clf_params)
	TLparams = get_TLparams(path_TL_params)
	
	final_result = runAnalise(clfParams, TLparams)
	print(final_result)
	# path_file = os.path.join(save_path,'final_results.json')
	# import json
	# with open(path_file, 'wb') as handle:
	# 	json.dump(final_result, handle)

