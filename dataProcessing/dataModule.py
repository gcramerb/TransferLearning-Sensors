from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split,Dataset
import os,random
import numpy as np
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

	def prepare_data(self):
	 # download
		file = os.path.join(self.data_dir, f'{self.source}_f25_t2.npz')
		with np.load(file, allow_pickle=True) as tmp:
			self.sourceX = tmp['X']
			y = tmp['y']
		y = categorical_to_int(y).astype('int')
		self.sourceY = np.argmax(y, axis=1).astype('long')
		file = os.path.join(self.data_dir, f'{self.target}_f25_t2.npz')
		with np.load(file, allow_pickle=True) as tmp:
			self.targetX = tmp['X']
			y = tmp['y']
		y = categorical_to_int(y).astype('int')
		self.targetY = np.argmax(y, axis=1).astype('long')
		self.dataset = crossDataset((self.sourceX,self.sourceY),(self.targetX, self.targetY))

	def setup(self, stage=None,rate = 0.1):
        # Assign train/val datasets for use in dataloaders
		if stage == "fit" or stage is None:
			nSamples = len(self.dataset)
			valL = int(nSamples*rate)
			trainL = nSamples - valL
			self.data_train, self.data_val = random_split(self.dataset, [trainL,valL], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
		if stage == "test" or stage is None:
			self.data_test = self.dataset

	def train_dataloader(self):
		return DataLoader(
			self.dataset,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			)

	def val_dataloader(self):
		return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)