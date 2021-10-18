from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split,Dataset
import os,random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from Utils.data import categorical_to_int


class crossDataset(Dataset):
	def __init__(self,source,target):
		s,ys = source
		t,yt = target
		data = []
		for i in range(len(s)):
			data.append((s[i], 0, ys[i]))
		for i in range(len(t)):
			data.append((t[i], 1, yt[i]))
		self.dataset = data
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		sample = {'data': self.dataset[idx][0], 'domain': self.dataset[idx][1], 'label': self.dataset[idx][2]}
		return sample
	
	
class CrossDatasetModule(LightningDataModule):
	def __init__(
		self,
		data_dir: str = None,
		source: str = "Dsads",
		target: str = "Ucihar",
		batch_size: int = 128,
		num_workers: int = 1,
	):
		super().__init__()
		self.data_dir = data_dir
		self.source  = source
		self.target = target
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.num_classes = 6


	def setup(self,stage = None,valRate = 0.1,testRate = .1):

		file = os.path.join(self.data_dir, f'{self.source}_f25_t2.npz')
		with np.load(file, allow_pickle=True) as tmp:
			self.sourceX = tmp['X'].astype('float32')
			y = tmp['y']
		y = categorical_to_int(y).astype('int')
		
		self.sourceY = np.argmax(y, axis=1).astype('long')
		n_classSou = len(pd.unique(self.sourceY))
		file = os.path.join(self.data_dir, f'{self.target}_f25_t2.npz')
		with np.load(file, allow_pickle=True) as tmp:
			self.targetX = tmp['X'].astype('float32')
			y = tmp['y']
		y = categorical_to_int(y).astype('int')
		self.targetY = np.argmax(y, axis=1).astype('long')
		n_classTarg = len(pd.unique(self.targetY))
		if n_classTarg ==n_classSou:
			self.num_classes = n_classTarg
		else:
			raise ValueError("Incompatible classes" )

		self.dataset = crossDataset((self.sourceX, self.sourceY), (self.targetX, self.targetY))
		nSamples = len(self.dataset)
		valLen = int(nSamples * valRate)
		testLen = int(nSamples * testRate)
		trainL = nSamples - testLen
		self.dataTrain, self.dataTest = random_split(self.dataset, [trainL, testLen], generator=torch.Generator().manual_seed(42))
		trainL = trainL - valLen
		self.dataTrain, self.dataVal = random_split(self.dataTrain, [trainL, valLen], generator=torch.Generator().manual_seed(0))
	
	def train_dataloader(self):
		return DataLoader(
			self.dataTrain,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			)

	def val_dataloader(self):
		return DataLoader(self.dataVal, batch_size=self.batch_size, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.dataTest, batch_size=self.batch_size, num_workers=self.num_workers)