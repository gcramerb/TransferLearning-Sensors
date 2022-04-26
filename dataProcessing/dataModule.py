from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

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
			file_name:str = None,
			n_classes: int = 4,
			input_shape: tuple = (1,50,6),
			batch_size: int = 128,
			type: str = 'source',
			num_workers: int = 1,
	):
		super().__init__()
		self.data_dir = data_dir
		self.datasetName = datasetName
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.n_classes = n_classes
		self.input_shape = input_shape
		self.type = type
		self.X_val = None

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

	def setup(self,split = True,normalize = False, fold_i = None,SL_path_file = None):
		valRate = 0.1
		testRate = 0.2
		if self.datasetName.split('.')[-1] =='npz':
			is_cat = False
			file = os.path.join(self.data_dir,self.datasetName)
		else:
			is_cat = True
			file = os.path.join(self.data_dir, f'{self.datasetName}_f25_t2_{self.n_classes}actv.npz')
		
		with np.load(file, allow_pickle=True) as tmp:
			X = tmp['X'].astype('float32')
			Y = tmp['y']
			#self.folds = tmp['folds']
		if is_cat:
			y = categorical_to_int(Y).astype('int')
			Y = np.argmax(y, axis=1).astype('long')
		
		if fold_i is not None:
			raise ValueError("fold_i must be None")
			# self.X_val = X[self.folds[fold_i][1]]
			# self.X_train = X[self.folds[fold_i][0]]
			# self.Y_train = Y[self.folds[fold_i][0]]
			# self.Y_val = Y[self.folds[fold_i][1]]
		elif SL_path_file is not None:
			with np.load(SL_path_file, allow_pickle=True) as tmp:
				Xsl = tmp['dataSL'].astype('float32')
				ysl = tmp['ySL']
			self.X_val = X
			self.Y_val = Y
			self.Y_train = np.concatenate([Y,ysl],axis = 0)
			self.X_train = np.concatenate([X,Xsl],axis = 0)
			del self.folds
		else:
			self.X_val = X
			self.Y_val = Y
			self.X_train = X
			self.Y_train = Y

		if normalize:
			#TODO: normalizar os dados source e target juntos ?
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
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
	
	def val_dataloader(self):
		return DataLoader(
			self.dataVal,
	        batch_size=self.batch_size,
	        shuffle=True,
	        num_workers=self.num_workers,
	        drop_last=False)
	
	def test_dataloader(self):
		return DataLoader(self.dataTest,
		                  batch_size=self.batch_size,
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=False)


class CrossDatasetModule(LightningDataModule):
	def __init__(
			self,
			data_dir: str = None,
			sourceName: str = "Dsads",
			targetName: str = "Ucihar",
			norm: bool = True,
			n_classes: int = 6,
			input_shape: tuple = (1,50,6),
			batch_size: int = 128,
			num_workers: int = 1,
	):
		super().__init__()
		self.data_dir = data_dir
		self.sourceName = sourceName
		self.targetName = targetName
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.n_classes = n_classes
		self.input_shape = input_shape
		self.norm = norm
		
		self.dataTrain = {}
		self.dataVal = {}
		self.dataTest = {}
	
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
	
	def _setup(self,SL_path_file = None):
		self.setup_dataset(dataset =  self.sourceName,domain = 'source', SL_path_file=SL_path_file)
		self.setup_dataset(dataset = self.targetName,domain ='target')

	def setup_dataset(self,dataset, domain, SL_path_file=None):
		file = os.path.join(self.data_dir, f'{dataset}_f25_t2_{self.n_classes}actv.npz')
		with np.load(file, allow_pickle=True) as tmp:
			X = tmp['X'].astype('float32')
			y = tmp['y']
		y = categorical_to_int(y).astype('int')
		Y = np.argmax(y, axis=1).astype('long')

		if SL_path_file is not None:
			with np.load(SL_path_file, allow_pickle=True) as tmp:
				Xsl = tmp['Xsl'].astype('float32')
				ysl = tmp['ysl']

			X_val = X
			Y_val = Y
			Y_train = np.concatenate([Y, ysl], axis=0)
			X_train = np.concatenate([X, Xsl], axis=0)
		else:
			X_val = X
			Y_val = Y
			X_train = X
			Y_train = Y
		
		if self.norm:
			# TODO: normalizar os dados source e target juntos ?
			X_train, Y_train = self.normalize(X_train, Y_train)
			X_val, Y_val = self.normalize(X_val, Y_val)
		
		if self.input_shape[0] == 2:
			X_train = np.concatenate([X_train[:, :, :, 0:3], X_train[:, :, :, 3:6]], axis=1)
			X_val = np.concatenate([X_val[:, :, :, 0:3], X_val[:, :, :, 3:6]], axis=1)
		
		self.dataTrain[domain] = myDataset(X_train, Y_train)
		self.dataVal[domain] = myDataset(X_val, Y_val)
		self.dataTest[domain] = myDataset(X_val, Y_val)
		

	def train_dataloader(self):
		source_loader =  DataLoader(
			self.dataTrain['source'],
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
		target_loader =  DataLoader(
			self.dataTrain['target'],
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
		return [source_loader,target_loader]

	def val_dataloader(self):
		source_loader =  DataLoader(
			self.dataVal['source'],
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
		target_loader =  DataLoader(
			self.dataVal['target'],
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
		return [source_loader,target_loader]

	def test_dataloader(self):
		source_loader =  DataLoader(
			self.dataTest['source'],
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
		target_loader =  DataLoader(
			self.dataTest['target'],
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
		return [source_loader,target_loader]