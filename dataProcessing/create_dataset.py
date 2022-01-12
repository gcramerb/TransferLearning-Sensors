import os,random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from Utils.data import categorical_to_int
from torch.utils.data import Dataset



def getData(inPath,dataset,getLabel,categoricalLab = False,valRate = 0.0):
	file = os.path.join(inPath, f'{dataset}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	if not categoricalLab:
		y = categorical_to_int(y).astype('int')
		y = np.argmax(y, axis=1)
	#X = np.transpose(X.astype('float32'), (0, 2, 3, 1))
	valIdx = random.sample(range(0, len(X)), int(len(X)*valRate))
	maskVal = np.array([False] *len(X))
	maskVal[valIdx] = True
	maskTrain = np.logical_not(maskVal)
	data, dataVal = X[maskTrain, :, :, 0:6], X[maskVal, :, :, 0:6]
	y,yVal = y[maskTrain],y[maskVal]
	if getLabel and valRate > .0:
		return (data,y), (dataVal,yVal)
	elif getLabel and valRate == .0:
		return data, y
	else:
		return (data,None), (dataVal,None)
	
	
