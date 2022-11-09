from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler

import os
import numpy as np
import torch
from Utils.data import categorical_to_int
# from models.pseudoLabSelection import generate_pseudoLab

class myDataset(Dataset):
	def __init__(self, X, Y,norm = False):
		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.Y)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		if False:
			return {'data': self.transform(torch.tensor(self.X[idx])), 'label': self.Y[idx]}

		return  {'data': self.X[idx], 'label': self.Y[idx]}


class SingleDatasetModule(LightningDataModule):
	def __init__(
			self,
			data_dir: str = None,
			datasetName: str = "Dsads",
			n_classes: int = 4,
			freq: int = 25,
			input_shape: tuple = (1,50,6),
			batch_size: int = 128,
			num_workers: int = 1,
			oneHotLabel: bool = False,
			shuffle: bool = False
	):
		super().__init__()
		self.data_dir = data_dir
		self.datasetName = datasetName
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.n_classes = n_classes
		self.freq = freq
		self.input_shape = input_shape
		self.oneHotLabel = oneHotLabel
		self.X_val = None
		self.shuffle = shuffle

	def normalize(self,X,Y):
		newX = []
		newY = []
		m = np.array([np.mean(X[:,0, :, i]) for i in range(X.shape[-1])])
		std = np.array([np.std(X[:,0, :, i]) for i in range(X.shape[-1])])
		for sample,label in zip(X[:,0,:,:],Y):
			sample = (sample - m)/std
			newX.append(sample)
			newY.append(label)
		newX, newY = np.array(newX), np.array(newY)
		return newX[:,None,:,:],newY

	def oneHotEncoding(self,label):
		labelOH = np.zeros((label.size, label.max() + 1))
		labelOH[np.arange(label.size), label] = 1
		return labelOH
	def setup(self,normalize = True, fold_i = None,SL_path_file = None, fileName = None):
		"""
		The data input is going to be always (None,1,50,6) = np(None,dumb,Freq,3*n_sensors)
		
		:param normalize:
		:param fold_i:
		:param SL_path_file:
		:return:
		"""
		if fileName is None:
			fileName =  f'{self.datasetName}_f{self.freq}_t2_{self.n_classes}actv.npz'
		
		file = os.path.join(self.data_dir,fileName)
		with np.load(file, allow_pickle=True) as tmp:
			X = tmp['X'].astype('float32')
			Y = tmp['y']
			self.folds = tmp['folds']
		
		if Y.dtype.type is np.str_:
			y = categorical_to_int(Y).astype('int')
			Y = np.argmax(y, axis=1).astype('long')
		if self.oneHotLabel:
			Y = self.oneHotEncoding(Y)
		if fold_i is not None:
			self.X_val = X[self.folds[fold_i][1]]
			self.X_train = X[self.folds[fold_i][0]]
			self.Y_train = Y[self.folds[fold_i][0]]
			self.Y_val = Y[self.folds[fold_i][1]]
		else:
			self.X_val = X
			self.Y_val = Y
			
			if SL_path_file is not None:
				print('\n There is a incesistency! \n')
				with np.load(SL_path_file, allow_pickle=True) as tmp:
					Xsl = tmp['X'].astype('float32')
					ysl = tmp['y']
				self.Y_train = np.concatenate([Y,ysl],axis = 0)
				self.X_train = np.concatenate([X,Xsl],axis = 0)
			else:
				self.X_train = X
				self.Y_train = Y

		if normalize:
			#TODO: normalizar os dados source e target juntos (no soft Labelling technique) ?
			self.X_train,self.Y_train = self.normalize(self.X_train,self.Y_train)
			self.X_val, self.Y_val = self.normalize(self.X_val, self.Y_val)
		
		if self.input_shape[0] == 2:
			self.X_train = np.concatenate([self.X_train[:,:,:,0:3],self.X_train[:,:,:,3:6]],axis =1)
			self.X_val = np.concatenate([self.X_val[:, :, :, 0:3], self.X_val[:, :, :, 3:6]], axis=1)
			
		self.dataTrain = myDataset(self.X_train, self.Y_train)
		self.dataVal = myDataset(self.X_val, self.Y_val)
		self.dataTest =myDataset(self.X_val, self.Y_val)
		
	def train_dataloader(self):
		return DataLoader(
			self.dataTrain,
			sampler=ImbalancedDatasetSampler(train_dataset),
			shuffle=self.shuffle,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
	
	def val_dataloader(self):
		return DataLoader(
			self.dataVal,
	        batch_size=self.batch_size,
	        shuffle=self.shuffle,
	        num_workers=self.num_workers,
	        drop_last=False)
	
	def test_dataloader(self):
		return DataLoader(
			self.dataTest,
          batch_size=self.batch_size,
          shuffle=self.shuffle,
          num_workers=self.num_workers,
          drop_last=False)


class MultiDatasetModule(LightningDataModule):
	def __init__(
			self,
			data_dir: str = None,
			datasetList: list = ["Dsads","Uschad","Pamap2"],
			n_classes: int = 6,
			freq: int = 50,
			input_shape: tuple = (2, 100, 3),
			batch_size: int = 128,
			num_workers: int = 1,
			oneHotLabel: bool = False,
			shuffle: bool = False
	):
		super().__init__()
		self.data_dir = data_dir
		self.datasetList = datasetList
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.n_classes = n_classes
		self.freq = freq
		self.input_shape = input_shape
		self.oneHotLabel = oneHotLabel
		self.X_val = []
		self.X_train = []
		self.Y_val = []
		self.Y_train = []
		self.shuffle = shuffle
	
	def normalize(self, X, Y):
		newX = []
		newY = []
		m = np.array([np.mean(X[:, 0, :, i]) for i in range(X.shape[-1])])
		std = np.array([np.std(X[:, 0, :, i]) for i in range(X.shape[-1])])
		for sample, label in zip(X[:, 0, :, :], Y):
			sample = (sample - m) / std
			newX.append(sample)
			newY.append(label)
		newX, newY = np.array(newX), np.array(newY)
		return newX[:, None, :, :], newY
	
	def oneHotEncoding(self, label):
		labelOH = np.zeros((label.size, label.max() + 1))
		labelOH[np.arange(label.size), label] = 1
		return labelOH
	
	def setup(self, normalize=True, fileName=None):
		"""
		The data input is going to be always (None,1,50,6) = np(None,dumb,Freq,3*n_sensors)

		:param normalize:
		:param fold_i:
		:param SL_path_file:
		:return:
		"""
		for datasetName in self.datasetList:
			fileName = f'{datasetName}_f{self.freq}_t2_{self.n_classes}actv.npz'
			file = os.path.join(self.data_dir, fileName)
			with np.load(file, allow_pickle=True) as tmp:
				X = tmp['X'].astype('float32')
				Y = tmp['y']
			
			if Y.dtype.type is np.str_:
				y = categorical_to_int(Y).astype('int')
				Y = np.argmax(y, axis=1).astype('long')
			if self.oneHotLabel:
				Y = self.oneHotEncoding(Y)
			X_val = X
			Y_val = Y
			X_train = X
			Y_train = Y
			
			if normalize:
				# TODO: normalizar os dados source e target juntos (no soft Labelling technique) ?
				X_train, Y_train = self.normalize(X_train, Y_train)
				X_val, Y_val = self.normalize(X_val, Y_val)
			
			if self.input_shape[0] == 2:
				X_train = np.concatenate([X_train[:, :, :, 0:3], X_train[:, :, :, 3:6]], axis=1)
				X_val = np.concatenate([X_val[:, :, :, 0:3], X_val[:, :, :, 3:6]], axis=1)
			self.X_train.append(X_train)
			self.X_val.append(X_val)
			self.Y_train.append(Y_train)
			self.Y_val.append(Y_val)
		
		self.dataTrain = myDataset(np.concatenate(self.X_train,axis = 0),np.concatenate( self.Y_train,axis = 0))
		self.dataVal = myDataset(np.concatenate(self.X_val,axis = 0), np.concatenate(self.Y_val,axis = 0))
		self.dataTest = myDataset(np.concatenate(self.X_val,axis = 0),np.concatenate( self.Y_val,axis = 0))
	
	def train_dataloader(self):
		return DataLoader(
			self.dataTrain,
			shuffle=self.shuffle,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
	
	def val_dataloader(self):
		return DataLoader(
			self.dataVal,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			num_workers=self.num_workers,
			drop_last=False)
	
	def test_dataloader(self):
		return DataLoader(
			self.dataTest,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			num_workers=self.num_workers,
			drop_last=False)