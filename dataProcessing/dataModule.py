from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split, Dataset

import os, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from Utils.data import categorical_to_int

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
			n_classes: int = 6,
			inputShape: tuple = (1,50,6),
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
		self.inputShape = inputShape
		self.type = type
		#self.transform = transforms.Normalize(0, 1, inplace=False)
	def get_n_folds(self):
		return range(len(self.folds))
	
	def set_fold(self,fold_idx):
		test_idx = self.folds[fold_idx][1]
		train_idx = self.folds[fold_idx][0]

		self.dataTrain = myDataset(self.X[train_idx], self.Y[train_idx])
		self.dataVal =  myDataset(self.X[test_idx], self.Y[test_idx])
		self.dataTest =  myDataset(self.X[test_idx], self.Y[test_idx])
		
	def normalize(self):
		newX = []
		newY = []
		tol = 3
		m = np.array([np.mean(self.X[:, 0, :, i]) for i in range(self.X.shape[-1])])
		std = np.array([np.std(self.X[:,0,:,i]) for i in range(self.X.shape[-1])])
		out_up = m + tol*std
		out_down = m - tol*std

		for sample,label in zip(self.X[:,0,:,:],self.Y):
			#check if this sample is an outlier:
			min = np.min(sample[:,:],axis = 0)
			max = np.max(sample[:,: ],axis = 0)
			if ((out_up < max) + 0).sum() + ((out_down > min) + 0).sum() == 0:
				# normalization [-0.5, 0.5]
				new_sample = (sample - min)/(max-min) - 0.5
				newX.append(new_sample[None,:,:])
				newY.append(label)
		return np.array(newX),np.array(newY)

	def setup(self, stage=None, valRate=0.1, testRate=.2,Loso = False,split = True,normalize = False):
		if normalize and Loso:
			raise ValueError("This functionality not implemented! ")
		file = os.path.join(self.data_dir, f'{self.datasetName}_f25_t2_{self.n_classes}actv.npz')
		with np.load(file, allow_pickle=True) as tmp:
			self.X = tmp['X'].astype('float32')
			y = tmp['y']
			self.folds = tmp['folds']
		# if self.datasetName =='Pamap2':
		#
		# 	acts = ['Pamap2-walking', 'Pamap2-ascending stairs', 'Pamap2-descending stairs', 'Pamap2-lying']
		# 	idx = [i for i, v in enumerate(y) if v in acts]
		# 	self.X = self.X[idx].copy()
		# 	y = y[idx].copy()

		#self.n_classes = len(pd.unique(self.Y))
		y = categorical_to_int(y).astype('int')
		self.Y = np.argmax(y, axis=1).astype('long')
		if normalize:
			self.X,self.Y = self.normalize()
			
		if self.inputShape[0] == 2:
			self.X = np.concatenate([self.X[:,:,:,0:3],self.X[:,:,:,3:6]],axis =1)
		self.dataset = myDataset(self.X, self.Y)


		if not Loso and split:
			nSamples = len(self.dataset)
			valLen = int(nSamples * valRate)
			testLen = int(nSamples * testRate)
			trainL = nSamples - testLen
			self.dataTrain, self.dataTest = random_split(self.dataset, [trainL, testLen],
			                                             generator=torch.Generator().manual_seed(42))
			trainL = trainL - valLen
			self.dataTrain, self.dataVal = random_split(self.dataTrain, [trainL, valLen],
			
	                                            generator=torch.Generator().manual_seed(0))
			if self.type == 'target':
				self.dataVal = self.dataset
				self.dataTest = self.dataset
		
		
		elif not split:
			self.dataTrain = self.dataset
			self.dataVal = self.dataset
			self.dataTest = self.dataset
			


	def dataloader(self):
		return DataLoader(
			self.dataset,
			shuffle=True,
			batch_size=len(self.dataset),
			num_workers=self.num_workers,
			drop_last=True)
		
	def train_dataloader(self):
		return DataLoader(
			self.dataTrain,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
	
	def val_dataloader(self):
		return DataLoader(self.dataVal,
		                  batch_size=self.batch_size,
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=True)
	
	def test_dataloader(self):
		return DataLoader(self.dataTest,
		                  batch_size=self.batch_size,
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=True)


class myCrossDataset(Dataset):
	def __init__(self, data, norm=False):
		source, target = data
		self.Xsource, self.Ysource = source
		self.Xtarget, self.Ytarget = target

		if norm:
			self.mean = (np.mean(X[:, 0, :, :]), np.mean(X[:, 1, :, :]))
			self.std = (np.std(X[:, 0, :, :]), np.std(X[:, 1, :, :]))
			self.transform = transforms.Normalize(self.mean, self.std)
		# transforms.Compose([transforms.ToTensor(),
		else:
			self.transform = None
	
	def __len__(self):
		return min(len(self.Ysource),len(self.Ytarget))

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return {'source': (self.Xsource[idx],self.Ysource[idx]), 'target': (self.Xtarget[idx],self.Ytarget[idx])}



class CrossDatasetModule(LightningDataModule):
	def __init__(
			self,
			data_dir: str = None,
			sourceName: str = "Dsads",
			targetName: str = "Ucihar",
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
		self.num_classes = n_classes
		self.input_shape = input_shape
		self.transform = transforms.Normalize(0, 1, inplace=False)
	
	def setup(self, stage=None, valRate=0.1, testRate=.2, Loso=False):
		file = os.path.join(self.data_dir, f'{self.sourceName}_f25_t2_{self.num_classes}actv.npz')
		with np.load(file, allow_pickle=True) as tmp:
			Xsource = tmp['X'].astype('float32')
			y = tmp['y']
			foldsSource = tmp['folds']
		# self.n_classes = len(pd.unique(self.Y))
		y = categorical_to_int(y).astype('long')
		Ysource = np.argmax(y, axis=1).astype('long')
		
		file = os.path.join(self.data_dir, f'{self.targetName}_f25_t2_{self.num_classes}actv.npz')
		with np.load(file, allow_pickle=True) as tmp:
			Xtarget = tmp['X'].astype('float32')
			y = tmp['y']
			foldsTarget = tmp['folds']
		# self.n_classes = len(pd.unique(self.Y))
		y = categorical_to_int(y).astype('long')
		Ytarget = np.argmax(y, axis=1).astype('long')

		if self.input_shape[-1] == 3:
			Xtarget = np.concatenate([Xtarget[:, :, :, 0:3], Xtarget[:, :, :, 3:6]], axis=1)
			Xsource = np.concatenate([Xsource[:, :, :, 0:3], Xsource[:, :, :, 3:6]], axis=1)
		
		if Loso:
			foldTestSource, foldValSource = np.random.randint(len(foldsSource), size=2)
			testIdxSource = foldsSource[foldTestSource][1]
			valIdxSource = foldsSource[foldValSource][1]
			trainIdxSource = list(set(foldsSource[foldTestSource][0]) - set(foldsSource[foldValSource][1]))
			
			foldTestTarget, foldValTarget = np.random.randint(len(foldsTarget), size=2)
			testIdxTarget = foldsTarget[foldTestTarget][1]
			valIdxTarget = foldsTarget[foldValTarget][1]
			trainIdxTarget = list(set(foldsTarget[foldTestTarget][0]) - set(foldsTarget[foldValTarget][1]))

			train = (Xsource[trainIdxSource],Ysource[trainIdxSource]),(Xtarget[trainIdxTarget],Ytarget[trainIdxTarget])
			val = (Xsource[valIdxSource],Ysource[valIdxSource]),(Xtarget[valIdxTarget],Ytarget[valIdxTarget])
			test = (Xsource[testIdxSource],Ysource[testIdxSource]),(Xtarget[testIdxTarget],Ytarget[testIdxTarget])
			
			self.dataTrain, self.dataVal, self.dataTest = myCrossDataset(train), myCrossDataset(val), myCrossDataset(test)

		else:
			source = Xsource,Ysource
			target = Xtarget,Ytarget
			data = source,target
			dataset = myCrossDataset(data)
			nSamples = len(dataset)
			valLen = int(nSamples * valRate)
			testLen = int(nSamples * testRate)
			trainL = nSamples - testLen
			self.dataTrain, self.dataTest = random_split(dataset, [trainL, testLen],
			                                             generator=torch.Generator().manual_seed(42))
			trainL = trainL - valLen
			self.dataTrain, self.dataVal = random_split(self.dataTrain, [trainL, valLen],
			                                            generator=torch.Generator().manual_seed(0))

	def train_dataloader(self):
		return DataLoader(
			self.dataTrain,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
	
	def val_dataloader(self):
		return DataLoader(self.dataVal,
		                  batch_size=len(self.dataVal),
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=True)
	
	def test_dataloader(self):
		return DataLoader(self.dataTest,
		                  batch_size=512,
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=True)