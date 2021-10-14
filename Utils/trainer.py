import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim


import sys, os, argparse
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, f1_score


sys.path.insert(0, '../')

from models.classifier import classifier
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss,OTLoss
#import geomloss


from dataProcessing.create_dataset import crossDataset, targetDataset, getData


class Trainer:
	def __init__(self, hyp=None):
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		
		if hyp['model'] == 'clf':
			self.model = classifier(n_class=6)
			self.loss = torch.nn.CrossEntropyLoss()
		elif hyp['model'] =='AE':
			self.model = ConvAutoencoder(hyp['model_hyp'])
			self.loss = torch.nn.MSELoss()
		self.model.build()
		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=hyp['lr'])
		self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
		

		
		if hyp['penalty'] == 'mmd':
			self.penalty = MMDLoss()
		elif hyp['penalty'] == 'ot':
			self.penalty = OTLoss()
	
	# from torchsummary import summary
	# summary(self.model, (1, 50, 6))
	
	def configTrain(self, alpha=0.8, n_ep=120, bs=128):
		self.bs = bs
		self.alpha = alpha
		self.epochs = n_ep
	
	def train(self, dataTrain,dataVal = None, printGrad = False):
		# transform = transforms.Compose([transforms.Resize(self.bs), transforms.ToTensor()])
		
		trainloader = DataLoader(dataTrain, shuffle=True, batch_size=self.bs)
		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss
			train_loss = 0.0
			main_loss = 0.0
			penalty_loss = 0.0
			for i, batch in enumerate(trainloader):
				data, domain, label = batch['data'], batch['domain'], batch['label']
				data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
				                                                                         dtype=torch.int), label.to(
					self.device, dtype=torch.long)
				#self.model.train()
				self.optimizer.zero_grad()
				latent, pred = self.model(data)
				if self.model.name == 'clf':
					sourceIdx = np.where(domain.cpu() == 0)[0]
					true = label[sourceIdx]
					pred = pred[sourceIdx]
				elif self.model.name == 'AE':
					true = data

				m_loss = self.loss(pred, true)
				p_loss = self.penalty(latent, domain)
				
				loss = self.alpha * m_loss + (1 - self.alpha) * p_loss
				loss.mean().backward()
				self.optimizer.step()
				train_loss += loss.mean().item()
				main_loss += m_loss.mean().item()
				penalty_loss += p_loss.mean().item()
			self.scheduler.step()
			train_loss = train_loss / len(trainloader)
			penalty_loss = penalty_loss / len(trainloader)
			main_loss = main_loss / len(trainloader)
			#print(train_loss, '  ', main_loss, '  ', penalty_loss, '\n')
			histTrainLoss.append(train_loss)
			if dataVal:
				valTarget,valSource,predValTarget,predValSource = self.predict(dataVal)
				accValSource = accuracy_score(valTarget,predValTarget)
				accValTarget = accuracy_score(valSource,predValSource)
				
				print('acc Val - source: ',accValSource,'target: ',accValTarget )

			if printGrad:
				print('Epoch:', '  ',epoch)
				for name, param in self.model.named_parameters():
					print(name, param.grad.mean())
				print('\n\n')
		return histTrainLoss
	
	def predict(self, dataTest):
		
		testloader = DataLoader(dataTest, shuffle=False, batch_size=len(dataTest))
		# PredSource = []
		# TrueSource = []
		# PredTarget = []
		# TrueTarget = []
		#
		with torch.no_grad():
			for i, batch in enumerate(testloader):
				data, domain, label = batch['data'], batch['domain'], batch['label']
				data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
				                                                                         dtype=torch.int), label.to(
					self.device, dtype=torch.long)
				latent, pred = self.model(data)
				sourceIdx = np.where(domain.cpu() == 0)[0]
				targetIdx = np.where(domain.cpu() != 0)[0]
				
				if self.model.name == 'clf':
					original = label.cpu().data.numpy()
					pred = np.argmax(pred.cpu().data.numpy(),axis = 1)

				elif self.model.name == 'AE':
					original = data.cpu().data.numpy()[0].astype('float')
					
				# TrueTarget.append(original[targetIdx])
				# TrueSource.append(original[sourceIdx])
				# PredSource.append(pred[sourceIdx])
				# PredTarget.append(pred[targetIdx])
			return original[targetIdx], original[sourceIdx], pred[targetIdx], pred[sourceIdx]
	
	def save(self, savePath):
		with open(savePath, 'w') as s:
			pickle.dump(self.model, s, protocol=pickle.HIGHEST_PROTOCOL)
	
	def loadModel(self, filePath):
		with open(filePath, 'rb') as m:
			self.model = pickle.load(m)
	
	def evaluate(self, data, dataRec, domain):
		mse_list = []
		mse_source = []
		mse_target = []
		source = data[np.where(domain == 0)[0]]
		sourceRec = dataRec[np.where(domain == 0)[0]]
		target = data[np.where(domain == 1)[0]]
		targetRec = dataRec[np.where(domain == 1)[0]]
		
		for k in range(data.shape[-1]):
			mse = np.square(np.subtract(data[:, :, k], dataRec[:, :, k])).mean(axis=1)
			mse_list.append(mse.mean())
			mseSource = np.square(np.subtract(source[:, :, k], sourceRec[:, :, k])).mean(axis=1)
			mse_source.append(mseSource.mean())
			mseTarget = np.square(np.subtract(targetRec[:, :, k], targetRec[:, :, k])).mean(axis=1)
			mse_target.append(mseTarget.mean())
		return np.mean(mse_list), np.mean(mse_source), np.mean(mse_target)




"""
class trainer:
	def __init__(self, hyp=None):
		self.model = ConvAutoencoder(hyp)
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.model.build()
		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
		self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
		self.penalty = MMDLoss()
		self.loss = torch.nn.MSELoss()
		#summary(self.model, (1, 50, 6))
		
	def configTrain(self, alpha=0.8, n_ep=80, bs=128):
		self.bs = bs
		self.alpha = alpha
		self.epochs = n_ep
		self.bs = bs

	def train(self, dataTrain):
		# transform = transforms.Compose([transforms.Resize(self.bs), transforms.ToTensor()])
		
		trainloader = DataLoader(dataTrain, shuffle=True, batch_size=self.bs)
		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss
			train_loss = 0.0
			main_loss = 0.0
			penalty_loss = 0.0
			for i, batch in enumerate(trainloader):
				data, domain, label = batch['data'], batch['domain'], batch['label']
				data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
				                                                                         dtype=torch.int), label.to(
					self.device, dtype=torch.int)
				self.optimizer.zero_grad()
				latent, pred = self.model(data)
				
				m_loss = self.loss(data, pred)
				p_loss = self.penalty(latent, domain)
				
				loss = self.alpha * m_loss + (1 - self.alpha) * p_loss
				loss.mean().backward()
				self.optimizer.step()
				train_loss += loss.mean().item()
				main_loss += m_loss.mean().item()
				penalty_loss += p_loss.mean().item()
			self.scheduler.step()
			train_loss = train_loss / len(trainloader)
			penalty_loss = penalty_loss / len(trainloader)
			main_loss = main_loss / len(trainloader)
			print(train_loss, '  ', mmd_train, '  ', mse_train, '\n')
			histTrainLoss.append(train_loss)
		return histTrainLoss
	
	def predict(self, xTest):
		
		testloader = DataLoader(xTest, shuffle=False, batch_size=len(xTest))
		dataRec = []
		dataLatent = []
		with torch.no_grad():
			for (i, batch) in testloader:
				data, domain, label = batch['data'], batch['domain'], batch['label']
				data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
				                                                                         dtype=torch.int), label.to(
					self.device, dtype=torch.int)
				# domain = domain.cpu().data.numpy()[0].astype('int')
				latent, rec = self.model(data)
				
				latent, rec = latent.cpu().data.numpy()[0], rec.cpu().data.numpy()[0]
				dataRec.append(rec)
				dataLatent.append(latent)
			return np.array(dataLatent), np.array(dataRec)
	
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
"""
