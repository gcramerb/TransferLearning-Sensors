import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from torchsummary import summary

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
		
		summary(self.model, (1, 50, 6))
		
		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
		self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.4)
		self.criterion = MMD_MSELoss()
		self.epochs = 70

	def train(self, sourceFile,targetFile, inPath):
		#transform = transforms.Compose([transforms.Resize(self.bs), transforms.ToTensor()])
		source = getData(inPath,sourceFile,True)
		
		target = getData(inPath, targetFile,False)
		dataTrain = crossDataset(source,target)
		trainloader = DataLoader(dataTrain, shuffle=True, batch_size=self.bs)

		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss
			train_loss = 0.0
			
			for i,batch  in enumerate(trainloader):
				data,domain,label = batch['data'],batch['domain'],batch['label']
				data,domain,label =  data.to(self.device,dtype=torch.float),domain.to(self.device,dtype=torch.int),label.to(self.device, dtype=torch.int)
				self.optimizer.zero_grad()
				latent,rec = self.model(data)
				loss = self.criterion(latent, domain,rec, data)
				loss.mean().backward()
				self.optimizer.step()
				train_loss += loss.mean().item()
			scheduler.step()
			train_loss = train_loss / len(trainloader)
			print(train_loss,'\n')
			histTrainLoss.append(train_loss)
		return histTrainLoss

	def predict(self, xTest):
		#TODO: rewrite this method
		testloader = DataLoader(xTest, shuffle=False, batch_size=len(xTest))
		dataRec = []
		with torch.no_grad():
			for (i,batch) in testloader:
				data,domain,label = batch['data'],batch['domain'],batch['label']
				data,domain,label = data.to(self.device,dtype=torch.float),domain.to(self.device,dtype=torch.int),label.to(self.device, dtype=torch.int)
				#domain = domain.cpu().data.numpy()[0].astype('int')
				latent,rec = self.model(data)
				latent, rec =  latent.cpu().data.numpy()[0],rec.cpu().data.numpy()[0]
				dataRec.append(rec)
			return np.array(dataRec)

	def save(self, savePath):
		with open(savePath, 'w') as s:
			pickle.dump(self.model, s, protocol=pickle.HIGHEST_PROTOCOL)
	
	def loadModel(self, filePath):
		with open(filePath, 'rb') as m:
			self.model = pickle.load(m)
	def evaluateRec(self,data,dataRec,domain):
		mse_list = []
		mse_source = []
		mse_target = []
		source = data[np.where(domain==0)[0]]
		sourceRec = dataRec[np.where(domain == 0)[0]]
		target = data[np.where(domain == 1)[0]]
		targetRec = dataRec[np.where(domain == 1)[0]]
		
		for k in range(data.shape[-1]):
			mse = np.square(np.subtract(data[:,:,k], dataRec[:,:,k])).mean(axis=1)
			mse_list.append(mse.mean())
			mseSource = np.square(np.subtract(source[:,:,k], sourceRec[:,:,k])).mean(axis=1)
			mse_source.append(mseSource.mean())
			mseTarget = np.square(np.subtract(targetRec[:,:,k], targetRec[:,:,k])).mean(axis=1)
			mse_target.append(mseTarget.mean())
		return np.mean(mse_list),np.mean(mse_source),np.mean(mse_target)
		
if __name__ == '__main__':
	AE = network()
	inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	AE.train('Dsads', 'Ucihar', inPath)


