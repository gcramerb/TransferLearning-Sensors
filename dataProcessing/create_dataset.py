import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from Utils.data import categorical_to_int
from torch.utils.data import Dataset

def getData(inPath,dataset,getLabel,categoricalLab = False):
	file = os.path.join(inPath, f'{dataset}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	if not categoricalLab:
		y = categorical_to_int(y).astype('int')
		y = np.argmax(y, axis=1)
	#X = np.transpose(X.astype('float32'), (0, 2, 3, 1))
	data = X[:, :, :, 0:6]
	if getLabel:
		return data,y
	else:
		return data,None
	
	

class crossDataset(Dataset):
	def __init__(self,source,target,n_class = 6,targetLab = False, transform = None):
		
		self.source,self.Ysource =source
		# self.Ysource= LabelEncoder().fit_transform(self.Ysource)
		self.Ysource = self.Ysource.astype('long')
		# self.Ysource = self.Ysource.reshape((len(self.Ysource), n_class))
		if targetLab:
			self.target,self.Ytarget = target
		else:
			self.target,_ = target
			self.Ytarget = -1*np.ones(len(self.target),dtype = 'int')
		self.transform = transform

	def processTorch(self,s,t,ys,yt):
		dataP = []
		for i in range(len(s)):
			dataP.append((s[i],0,ys[i]))
		for i in range(len(t)):
			dataP.append((t[i],1,yt[i]))
		return dataP

	def __len__(self):
		return len(self.source) + len(self.target)
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		dataset = self.processTorch(self.source,self.target,self.Ysource,self.Ytarget)
		sample = {'data':dataset[idx][0],'domain':dataset[idx][1],'label':dataset[idx][2]}
		return sample


class targetDataset(Dataset):
	def __init__(self,  target, transform=None):
		
		self.target, self.y = target

	
	def processTorch(self, t, y):
		dataP = []
		for i in range(len(t)):
			dataP.append((t[i], 1, self.y[i]))
		return dataP
	
	def __len__(self):
		return len(self.target)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		dataset = self.processTorch(self.target, self.y)
		sample = {'data': dataset[idx][0], 'domain': dataset[idx][1], 'label': dataset[idx][2]}
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