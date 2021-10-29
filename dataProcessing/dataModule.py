from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split, Dataset
from  torchvision import transforms
import os, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from Utils.data import categorical_to_int

class myDataset(Dataset):
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.Y)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		sample = {'data': self.X[idx],'label': self.Y[idx]}
		return sample


class CrossDatasetModule(LightningDataModule):
	def __init__(
			self,
			data_dir: str = None,
			datasetName: str = "Dsads",
			case: str = "Source",
			batch_size: int = 128,
			num_workers: int = 1,
	):
		super().__init__()
		self.data_dir = data_dir
		self.datasetName = datasetName
		self.case = case
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.num_classes = 6
		self.transform = transforms.Normalize(0, 1, inplace=False)
	
	def setup(self, stage=None, valRate=0.1, testRate=.2,Loso = False):
		file = os.path.join(self.data_dir, f'{self.datasetName}_f25_t2.npz')
		with np.load(file, allow_pickle=True) as tmp:
			X = tmp['X'].astype('float32')
			y = tmp['y']
			folds = tmp['folds']
		#self.num_classes = len(pd.unique(self.Y))
		y = categorical_to_int(y).astype('int')
		Y = np.argmax(y, axis=1).astype('long')
		
		if Loso:

			fold_test, fold_val = np.random.randint(len(folds), size=2)
			test_idx = folds[fold_test][1]
			val_idx = folds[fold_val][1]
			train_idx = list(set(folds[fold_test][0]) - set(folds[fold_val][1]))
			self.dataTrain,self.dataVal,self.dataTest = myDataset(X[train_idx],Y[train_idx]),myDataset(X[val_idx],Y[val_idx]),myDataset(X[test_idx],Y[test_idx])


		else:
			dataset = myDataset(X,Y)
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
		                  batch_size=len(self.dataTest),
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=True)