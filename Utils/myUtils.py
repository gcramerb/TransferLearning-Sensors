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
		TLparams['gan'] = TLparams['gan'] =='True'
		return TLparams
	TLparams = {}
	TLparams['lr'] =  0.001
	TLparams['gan'] = False
	TLparams['lr_gan'] = 0.0005
	TLparams['bs'] = 128
	TLparams['step_size'] = None
	TLparams['epoch'] = 50
	TLparams['feat_eng'] = 'sym'
	TLparams['alpha'] = 0.5
	TLparams['beta'] = 0.5
	TLparams['discrepancy'] = 'ot'
	TLparams['weight_decay'] = 0.1
	return TLparams


def get_SLparams(path_file=None):
	if path_file:
		with open(path_file) as f:
			SLparams = json.load(f)
		return SLparams
	SLparams = {}
	SLparams['lr'] = 0.001
	SLparams['bs'] = 128
	SLparams['step_size'] = None
	SLparams['epoch'] = 15
	SLparams['feat_eng'] = 'sym'
	SLparams['alpha'] = 0.75
	SLparams['discrepancy'] = 'ot'
	SLparams['weight_decay'] = 0.1
	SLparams['iter'] = 10
	SLparams['trashold'] = 0.49
	return SLparams
	
def get_Clfparams(path_file = None):
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = 10
	
	clfParams['epoch'] = 10
	clfParams["dropout_rate"] = 0.2
	clfParams['bs'] = 128
	clfParams['lr'] = 0.0008
	clfParams['weight_decay'] = 0.15
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
	

