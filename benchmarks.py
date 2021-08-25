"""
Run some simple methods of Transfer Learning
"""

import numpy as np
import pandas as pd
from enum import Enum
import scipy.stats as st
import csv, sys, glob, os, json,time

import argparse
from Dataset import Datasets
from Dataset.Ucihar import UCIHAR, SignalsUcihar,actNameUcihar
from Dataset.Dsads import DSADS, SignalsDsads, actNameDsads
from Dataset.Uschad import USCHAD, SignalsUschad, actNameUschad
from Dataset.Pamap2 import PAMAP2, SignalsPamap2, actNamePamap2
from Utils.actTranslate import actNameVersions
from Process.Manager import preprocess_datasets
from Process.Protocol import Loso

myActNames = {
	'walking': 0,
	'walking forward': 0,
	'ascending stairs': 1,
	'walking up': 1,
	'descending stairs': 2,
	'walking down': 2,
	'sitting':3,
	'standing':4,
	'lying':5,
	'lying on back':5,
	'lying on right':5,
	'laying':5,
	'sleeping':5
}


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--datasetTrain', type=str, default='Dsads')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
args = parser.parse_args()

if args.slurm:
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/")
	sys.path.insert(0, classifiersPath)
	from DCNNclassifier import DCNNclassifier
	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.7', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)
	

else:
	args.inPath = os.path.abspath('C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\')
	args.outPath  = os.path.realpath('results')
	
	classifiersPath = os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\")
	sys.path.insert(0,classifiersPath)
	from DCNNclassifier import DCNNclassifier

	
def processData():
	source = os.path.join(args.inPath, 'originals')
	outPath = os.path.join(args.inPath, 'frankDataset')
	
	# Creating datasets
	pamaFile = os.path.join(source, 'PAMAP2')
	p2 = PAMAP2('Pamap2', pamaFile, outPath, freq=100, trials_per_file=10000)
	sig_pm = [SignalsPamap2.acc1_chest_X, SignalsPamap2.acc1_chest_Y, SignalsPamap2.acc1_chest_Z]
	sig_pm += [SignalsPamap2.gyr_chest_X, SignalsPamap2.gyr_chest_Y, SignalsPamap2.gyr_chest_Z]
	p2.set_signals_use(sig_pm)
	datasets.append(p2)
	
	uscFile = os.path.join(source, 'USC-HAR')
	usc = USCHAD('Uschad', uscFile, outPath, freq=100, trials_per_file=10000)
	sig_usc = [SignalsUschad.acc_front_right_hip_X, SignalsUschad.acc_front_right_hip_Y,
	           SignalsUschad.acc_front_right_hip_Z]
	sig_usc += [SignalsUschad.gyr_front_right_hip_X, SignalsUschad.gyr_front_right_hip_Y,
	            SignalsUschad.gyr_front_right_hip_Z]
	usc.set_signals_use(sig_usc)
	datasets.append(usc)
	
	dsaFile = os.path.join(source, 'uci-daily-and-sports-activities')
	dsa = DSADS('Dsads', dsaFile, outPath, freq=25, trials_per_file=10000)
	sig_dsa = [SignalsDsads.acc_torso_X, SignalsDsads.acc_torso_Y, SignalsDsads.acc_torso_Z]
	sig_dsa += [SignalsDsads.gyr_torso_X, SignalsDsads.gyr_torso_Y, SignalsDsads.gyr_torso_Z]
	dsa.set_signals_use(sig_dsa)
	datasets.append(dsa)
	
	uciFile = os.path.join(source, 'uci-human-activity-recognition')
	uci = UCIHAR('Ucihar', uciFile, outPath, freq=50, trials_per_file=10000)
	sig_uci = [SignalsUcihar.acc_body_X, SignalsUcihar.acc_body_Y, SignalsUcihar.acc_body_Z]
	sig_uci += [SignalsUcihar.gyr_body_X, SignalsUcihar.gyr_body_Y, SignalsUcihar.gyr_body_Z]
	uci.set_signals_use(sig_uci)
	datasets.append(uci)
	
	preprocess_datasets(datasets)
	# # Creating Loso evaluate generating in differents datafiles (no merging dataset)
	selectedActs = ['Walking', 'Ascending stairs', 'Descending stairs', 'Standing', 'Sitting']
	for dat in datasets:
	# preprocessing
		generate_ev = Loso([dat], overlapping=0.5, time_wd=2)
	# Save name of dataset in variable y
		generate_ev.set_name_act()
		generate_ev.set_act_processed()
		generate_ev.remove_action(selectedActivities=selectedActs)
		generate_ev.simple_generate(outPath, new_freq=25)
def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
	return [m, ic[0],ic[1]]
def categorical_to_int(y):
	y1 = list(map(lambda x: x.split('-')[-1], y))
	y2 = np.array(list(map(lambda x: myActNames[x], y1)))
	y3 = np.zeros([len(y),len(pd.unique(y))])
	for i in range(len(y3)):
		y3[i,y2[i]] = 1
	return y3

def classification(datasetList,result):
	file = os.path.join(args.inPath, f'{args.datasetTrain}_f25_t2.npz')
	with np.load(file,allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
		folds = tmp['folds']
	aux = np.squeeze(X)
	X = np.expand_dims(aux, axis=-1)

	from sklearn.ensemble import RandomForestClassifier, VotingClassifier
	from sklearn.metrics import accuracy_score, recall_score, f1_score
	for fold_i in folds:
		Xtrain = X[fold_i[0]]
		Xtest = X[fold_i[1]]
		y_ = categorical_to_int(y)
		ytrain = y_[fold_i[0]]
		yTrue =y_[fold_i[1]]
		yTrue = np.argmax(yTrue, axis=1)
		classifier = DCNNclassifier()
		classifier.fit(Xtrain, ytrain)
		yPred = classifier.predict(Xtest)
		result[args.datasetTrain + '_acc'].append(accuracy_score(yTrue, yPred))
		result[args.datasetTrain + '_rec'].append(recall_score(yTrue, yPred, average='macro'))
		result[args.datasetTrain + '_f1'].append(f1_score(yTrue, yPred, average='macro'))
		for dat in datasetList:
			if dat != args.datasetTrain:
				file2 = os.path.join(args.inPath, f'{dat}_f25_t2.npz')
				with np.load(file2) as tmp:
					X2 = tmp['X']
					y2 = tmp['y']
				aux2 = np.squeeze(X2)
				X2 = np.expand_dims(aux2, axis=-1)
				yTrue2 = categorical_to_int(y2)
				yTrue2 = np.argmax(yTrue2, axis=1)
				yPred2 = classifier.predict(X2)
				result[dat + f'_{args.datasetTrain}_acc'].append(accuracy_score(yTrue2,yPred2))
				result[dat + f'_{args.datasetTrain}_rec'].append(recall_score(yTrue2,yPred2,average = 'macro'))
				result[dat + f'_{args.datasetTrain}_f1'].append(f1_score(yTrue2,yPred2,average = 'macro'))
	result = dict(map(lambda kv: (kv[0], myMetric(kv[1])), result.items()))
	outFile = os.path.join(args.outPath,f'simpleTL_{args.datasetTrain}.json')
	with open(outFile, 'w') as fp:
		json.dump(result, fp)
if __name__ == '__main__':
	
	#TODO
	# Experimentos:
	# 	Separar apenas as atividades que sao comuns
	# 	Separar os sensores Comuns.
	# 	Treinar A -> testar B
	# 	Treinar A + B -> testar B
	# 		Como que acessa os folds de teste provenientes apenas de B.

	#processData()
	result = {}
	datasetList = ['Dsads', 'Ucihar', 'Uschad', 'Pamap2']
	for dat in datasetList:
		if dat == args.datasetTrain:
			result[dat +'_acc'] = []
			result[dat + '_rec'] = []
			result[dat + '_f1'] = []
		else:
			result[dat + f'_{args.datasetTrain}_acc'] = []
			result[dat + f'_{args.datasetTrain}_rec'] = []
			result[dat + f'_{args.datasetTrain}_f1'] = []
	print('\n\n starting: ')
	print(args.datasetTrain,)
	print('\n\n\n')
	start = time.time()
	classification(datasetList,result)
	end = time.time()
	print("Time passed  = {}".format(end - start), flush=True)
	print('\n\n\n End   ')
	print(args.datasetTrain)
	print('\n\n\n')
	
	
	# file = os.path.join(args.inPath, f'{args.datasetTrain}_f25_t2.npz')
	# with np.load(file,allow_pickle=True) as tmp:
	# 	X = tmp['X']
	# 	y = tmp['y']
	# 	folds = tmp['folds']
	# print(np.isnan(np.sum(X)))
	# shp = X.shape
	# c = 0
	# for xi in range(shp[0]):
	# 	for axi in range(shp[3]):
	# 		if np.isnan(np.sum(X[xi,0,:,axi])):
	# 			c +=1
	# print(c)
	#
	# # file = os.path.join(args.inPath, f'{args.datasetTrain}_0.pkl')
	# # import pickle
	# # with open(file, "rb") as openfile:
	# # 	X = pickle.load(openfile)
	# # a= 0
	# # for k,v in X.items():
	# # 	for axi in range(v.shape[-1]):
	# # 		if np.isnan(np.sum(v[:, axi])):
	# # 			a += 1
	# # print(a)
