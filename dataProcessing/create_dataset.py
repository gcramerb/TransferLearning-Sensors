import os
import numpy as np
import torch
from Utils.data import categorical_to_int
from torch.utils.data import Dataset

def getData(inPath,dataset,getLabel):
	file = os.path.join(inPath, f'{dataset}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	Y = categorical_to_int(y).astype('int')
	Y = np.argmax(Y, axis=1)
	#X = np.transpose(X.astype('float32'), (0, 2, 3, 1))
	data = X[:, :, :, 0:6]
	if getLabel:
		return data,Y
	else:
		return data,None
	
	

class crossDataset(Dataset):
	def __init__(self,source,target,transform = None):
		
		self.source,self.Ysource =source
		self.target,_ = target
		self.transform = transform

	def processTorch(self,s,t,y):
		dataP = []
		r= np.min(len(s),len(t))
		
		for i in range(r):
			dataP.append((s[i],t[i],y[i]))
		return dataP

	def __len__(self):
		return len(self.source) + len(self.target)
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		dataset = self.processTorch(self.source,self.target,self.Ysource)
		sample = {'source':dataset[idx][0],'target':dataset[idx][1],'label':dataset[idx][2]}
		return sample


# inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
# sourceDat = 'Dsads'
# targetDat = 'Ucihar'
# file = os.path.join(inPath, f'{sourceDat}_f25_t2.npz')
# with np.load(file, allow_pickle=True) as tmp:
# 	X = tmp['X']
# 	y = tmp['y']
# Ysource = categorical_to_int(y).astype('int')
# Ysource = np.argmax(Ysource,axis = 1)
# X = np.transpose(X.astype('float32'),(0,2,3,1))
# source = X[:,:,0:6,:]
#
#
# file = os.path.join(inPath, f'{targetDat}_f25_t2.npz')
# with np.load(file, allow_pickle=True) as tmp:
# 	X = tmp['X']
# X = np.transpose(X.astype('float32'), (0, 2, 3, 1))
# target = X[:, :, 0:6, :]
#
# a = 1