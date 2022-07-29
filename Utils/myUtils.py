import numpy as np
import scipy.stats as st
import json,os

def MCI(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def get_TLparams(path_file = None):
	if path_file:
		with open(path_file) as f:
			TLparams = json.load(f)
		return TLparams
	TLparams = {}
	TLparams['lr'] =  0.001
	TLparams['bs'] = 128
	TLparams['step_size'] = None
	TLparams['epoch'] = 60
	TLparams['feat_eng'] = 'sym'
	TLparams['alpha'] = 0.5
	TLparams['beta'] = 0.001
	TLparams['discrepancy'] = 'ot'
	TLparams['weight_decay'] = 0.1
	return TLparams


def get_Stuparams(path_file=None):
	stuParams = {}
	stuParams['kernel_dim'] = [(5, 3), (25, 3)]
	stuParams['n_filters'] = (4, 16, 18, 24)
	stuParams['enc_dim'] = 64
	stuParams['input_shape'] = (2, 50, 3)
	if path_file:
		with open(path_file) as f:
			aux = json.load(f)
		for k,v in aux.items():
			stuParams[k] = v
		return stuParams

	stuParams['alpha'] = None
	stuParams['step_size'] = None
	stuParams['epoch'] = 12
	stuParams["dropout_rate"] = 0.2
	stuParams['bs'] = 128
	stuParams['lr'] = 0.0001
	stuParams['weight_decay'] = 0.2
	stuParams['iter'] = 10
	stuParams['trashold'] = 0.75
	return stuParams

	
def get_Clfparams(path_file = None):
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = None
	clfParams['clf_epoch'] = 12
	clfParams["dropout_rate"] = 0.2
	clfParams['bs'] = 128
	clfParams['clf_lr'] = 0.0001
	clfParams['weight_decay'] = 0.2
	if path_file:
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


	

