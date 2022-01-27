
def get_TLparams():
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
	
def get_Clfparams():
	clfParams = {}
	clfParams['kernel_dim'] = [(5, 3), (25, 3)]
	clfParams['n_filters'] = (4, 16, 18, 24)
	clfParams['enc_dim'] = 64
	clfParams['FE'] = 'fe2'
	clfParams['input_shape'] = (2, 50, 3)
	clfParams['alpha'] = None
	clfParams['step_size'] = None
	
	clfParams['epoch'] = 5
	clfParams["dropout_rate"] = 0.2
	clfParams['bs'] = 128
	clfParams['lr'] = 0.00005
	clfParams['weight_decay'] = 0.1
	return clfParams


def get_foldsInfo():
	folds = {}
	folds['Dsads'] = 8
	folds['Uschad'] = 14
	folds['Pamap2'] = 8
	folds['Ucihar'] = 30
	return folds
	

# def getModelHparams():
# 	clfParam = {}
# 	clfParam['kernel_dim'] = [(5, 3), (25, 1)]
# 	clfParam['n_filters'] = (6,8,24,24)
# 	clfParam['encDim'] =48
# 	clfParam["inputShape"] = (1,50,6)
#
# 	return clfParam
# def getTrainHparms():
# 	trainParams = {}
# 	trainParams['nEpoch'] = 30
# 	trainParams['batch_size'] = 64
# 	trainParams['alpha'] = 0.55
# 	trainParams['lr'] = 0.00015
# 	return trainParams
#
# def getHparamsPamap2():
# 	clfParam = {}
# 	clfParam['kernel_dim'] = [(5, 3), (25, 3)]
# 	clfParam['n_filters'] = (6,14,22,28)
# 	clfParam['encDim'] = 128
# 	clfParam["inputShape"] = (2,50,3)
# 	clfParam['DropoutRate'] = 0.2
# 	clfParam['FeName'] = 'fe1'
# 	trainParams = {}
# 	trainParams['nEpoch'] = 80
# 	trainParams['batch_size'] = 64
# 	trainParams['alpha'] = 0.0
# 	trainParams['lr'] = 0.00439
# 	trainParams['step_size'] = 15
# 	return clfParam,trainParams

