import numpy as np
import scipy.stats as st
import json,os

def MCI(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def getTeacherParams(path_file=None):
	Tparams = {}
	#
	# Tparams['n_filters'] = (4, 16, 18)
	# Tparams['enc_dim'] = 128
	Tparams['input_shape'] = (2, 50, 3)
	# Tparams['alpha'] = 0.5
	# Tparams['beta'] = 0.05
	# Tparams['epoch'] = 32
	Tparams["dropout_rate"] = 0.2
	Tparams['bs'] = 128
	Tparams['lr'] = 0.001
	Tparams['weight_decay'] = 0.2
	Tparams['iter'] = 10
	Tparams['trashold'] = 0.75
	Tparams['discrepancy'] = 'ot'
	
	if path_file is not None:
		with open(path_file) as f:
			aux = json.load(f)
		for k,v in aux.items():
			Tparams[k] = v
	if "f1" in Tparams.keys():
		Tparams['n_filters'] = (Tparams['f1'],Tparams['f2'],Tparams['f3'])
	if "kernel_dim" in Tparams.keys():
		Tparams['kernel_dim'] =[(5, 3), (Tparams['kernel_dim'], 3)]
	else:
		Tparams['kernel_dim'] = [(5, 3), (25, 3)]
	return Tparams

def getStudentParams(path_file=None):
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	#clfParams['input_shape'] = (2, 50, 3)
	clfParams['enc_dim'] = 64
	clfParams['alpha'] = 0.5
	clfParams['step_size'] = None
	clfParams['epoch'] = 70
	clfParams["dropout_rate"] = 0.2
	clfParams['lr'] = 0.001
	clfParams['weight_decay'] = 0.2
	if path_file is not None:
		with open(path_file) as f:
			aux = json.load(f)
		for k,v in aux.items():
			clfParams[k] = v
	return clfParams


def get_foldsInfo():
	folds = {}
	folds['Dsads'] = 8
	folds['Uschad'] = 14
	folds['Pamap2'] = 8
	folds['Ucihar'] = 30
	return folds

def getPLS_params():
	selectionParams = []
	for k in [8,64,128,256,512,1024,2056]:
		for l in [0.5,0.6,0.7,0.8,0.85]:
				for n in [10,20, 30,40,50,100,200]:
					param_i = {}
					param_i['nClusters'] = k
					param_i['labelConvergence'] = l
					param_i['minSamples'] = n
					selectionParams.append(param_i)
	return selectionParams
	# auxSelectionParams['params'] ={'nClusters':64 , 'labelConvergence':0.9 , 'minSamples':100}
	# bestParms = {}
	# bestParms['Ucihar_Pamap2'] = selectionParams
	# auxSelectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 50}
	# bestParms['Ucihar_Dsads'] =selectionParams
	# Uci_usc: ok
	# auxSelectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.6, 'minSamples': 100}
	# bestParms['Dsads_Pamap2'] =selectionParams
	# auxSelectionParams['params'] = {'nClusters': 32, 'labelConvergence':0.85, 'minSamples': 150}
	# bestParms['Dsads_Uschad'] =selectionParams
	# auxSelectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.6, 'minSamples': 100}
	# bestParms['Uschad_Dsads'] =selectionParams
	# auxSelectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 50}
	# bestParms['Uschad_Ucihar'] =selectionParams
	# auxSelectionParams['params'] = {'nClusters': 64, 'labelConvergence':0.9, 'minSamples': 50}
	# bestParms['Pamap2_Dsads'] = ok - need improovment
	# bestParms['Pamap2_Ucihar'] = Ok

