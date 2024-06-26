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
	TLparams['epoch'] = 7
	TLparams['feat_eng'] = 'sym'
	TLparams['alpha'] = 0.15
	TLparams['beta'] = 0.5
	TLparams['discrepancy'] = 'ot'
	TLparams['weight_decay'] = 0.1
	
	#only in soft-Label techinique:
	TLparams['iter'] = 5
	return TLparams
	
def get_Clfparams(path_file = None):
	
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['input_shape'] = (2, 250, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = 10
	
	clfParams['epoch'] = 10
	clfParams["dropout_rate"] = 0.25
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
	

