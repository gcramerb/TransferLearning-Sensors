import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim

# from torchsummary import summary

import sys, os,argparse
import numpy as np


from sklearn.metrics import accuracy_score, recall_score, f1_score
# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from models.classifier import classifier
from models.mmdLoss import myMMDLoss

from dataProcessing.create_dataset import crossDataset,targetDataset, getData


class trainer:
	def __init__(self, hyp=None):
		self.model = classifier(n_class = 6)
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.model.build()
		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
		self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
		self.penalty = myMMDLoss()
		self.loss = torch.nn.CrossEntropyLoss()
	
	# summary(self.model, (1, 50, 6))
	
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
					self.device, dtype=torch.long)
				self.optimizer.zero_grad()
				latent, pred = self.model(data)
				
				sourceIdx = np.where(domain.cpu()==0)[0]
				sourceLab = label[sourceIdx]
				sourcePred = pred[sourceIdx]
				m_loss = self.loss(sourcePred,sourceLab)
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
			print(train_loss, '  ', main_loss, '  ', penalty_loss, '\n')
			histTrainLoss.append(train_loss)
		return histTrainLoss
	
	def predict(self, xTest):
		
		testloader = DataLoader(xTest, shuffle=False, batch_size=len(xTest))
		yPredSource = []
		yTrueSource = []
		yPredTarget = []
		yTrueTarget = []

		with torch.no_grad():
			for (i, batch) in testloader:
				data, domain, label = batch['data'], batch['domain'], batch['label']
				data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
				                                                                         dtype=torch.int), label.to(
					self.device, dtype=torch.long)
				latent, pred = self.model(data)
				y = label.cpu().data.numpy()[0].astype('int')
				sourceIdx = np.where(domain.cpu() == 0)[0]
				targetIdx =  np.where(domain.cpu() != 0)[0]
				
				pred = np.argmax(pred.cpu().data.numpy()[0])
				yTrueSource.append(y[sourceIdx])
				yPredSource.append(pred[sourceIdx])
				yTrueTarget.append(y[targetIdx])
				yPredTarget.append(pred[targetIdx])

			return  np.array(yTrueSource),np.array(yPredSource) ,np.array(yTrueTarget),np.array(yPredTarget)
	
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


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--source', type=str, default="Dsads")
parser.add_argument('--target', type=str, default="Ucihar")
args = parser.parse_args()

if __name__ == '__main__':
	
	clf = trainer()
	clf.configTrain(bs=256)
	
	if args.inPath is None:
		args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
	source = getData(args.inPath, args.source, True)
	target = getData(args.inPath, args.target, False)
	dataTrain = crossDataset(source, target)
	clf.train(dataTrain)
	source = getData(args.inPath, args.source, True)
	target = getData(args.inPath, args.target, True)
	dataTest = crossDataset(source, target)
	yTrueSource,yPredSource, yTrueTarget,yPredTarget = clf.predict(dataTest)
	
	print('Source: ')
	print('\n',accuracy_score(yTrueSource,yPredSource),'\n')
	print(recall_score(yTrueSource,yPredSource,average = 'macro'),'\n')
	print(f1_score(yTrueSource,yPredSource,average = 'macro'),'\n')

	print('Target: ')
	print('\n',accuracy_score(yTrueTarget,yPredTarget),'\n')
	print(recall_score(yTrueTarget,yPredTarget,average = 'macro'),'\n')
	print(f1_score(yTrueTarget,yPredTarget,average = 'macro'),'\n')

