import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

import sys,os
from geomloss import SamplesLoss
sys.path.insert(0,'../')

from models.autoencoder import ConvAutoencoder
from models.mmdLoss import MMD_MSELoss
from dataProcessing.create_dataset import crossDataset,getData

class network:
	def __init__(self, bs=32,hyp = None):
		self.bs = bs
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.model = ConvAutoencoder(hyp).to(self.device)
		self.model.build()
		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
		self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.4)
		self.criterion = MMD_MSELoss()
		self.epochs = 70

	def train(self, sourceFile,targetFile, inPath):
		#transform = transforms.Compose([transforms.Resize(self.bs), transforms.ToTensor()])
		source = getData(inPath,sourceFile,True)
		target = getData(inPath, targetFile,False)
		dataTrain = crossDataset([source,target])
		trainloader = DataLoader(dataTrain, shuffle=True, batch_size=self.bs)

		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss
			train_loss = 0.0
			
			for i,batch  in enumerate(trainloader):
				source,target,label = batch['source'],batch['target'],batch['label']
				source,target,label= source.to(self.device,dtype=torch.float),target.to(self.device,dtype=torch.int),label.to(self.device, dtype=torch.int)
				self.optimizer.zero_grad()
				latent,rec = self.model(data)
				loss = self.criterion(latent,domain, rec, data)
				loss.mean().backward()
				self.optimizer.step()
				train_loss += loss.mean().item()
			scheduler.step()
			train_loss = train_loss / len(trainloader)
			histTrainLoss.append(train_loss)
		return histTrainLoss

	def predict(self, xTest):
		#TODO: rewrite this method
		testloader = DataLoader(xTest, shuffle=False, batch_size=len(xTest))

		with torch.no_grad():
			for (i,data) in testloader:
				acc, gyr, domain = data
				acc, gyr, domain = acc.to(self.device, dtype=torch.float), gyr.to(self.device,dtype=torch.float), domain.to(self.device, dtype=torch.int)
				
				domain = domain.cpu().data.numpy()[0].astype('int')

				encoded, dataRec = self.model([acc,gyr])
				encoded, dataRec =  encoded.cpu().data.numpy()[0],dataRec.cpu().data.numpy()[0]
			return encoded, dataRec

	def save(self, savePath):
		with open(savePath, 'w') as s:
			pickle.dump(self.model, s, protocol=pickle.HIGHEST_PROTOCOL)
	
	def loadModel(self, filePath):
		with open(filePath, 'rb') as m:
			self.model = pickle.load(m)
			
if __name__ == '__main__':
	AE = network()
	inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	AE.train('Dsads', 'Ucihar', inPath)


