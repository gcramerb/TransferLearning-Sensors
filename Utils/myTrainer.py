import torch
import torch.nn as nn
from torch.nn.functional import  cross_entropy
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch import optim
# seed = 19
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier,classifierTest
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss, OTLoss, classDistance
# import geomloss



from dataProcessing.create_dataset import crossDataset, targetDataset, getData
from dataProcessing.dataModule import CrossDatasetModule

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
from models.customLosses import MMDLoss, OTLoss
# import geomloss
from Utils.trainingConfig import EarlyStopping
from dataProcessing.create_dataset import crossDataset, targetDataset, getData


class myTrainer:
	def __init__(self, modelName, hypModel=None):
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.name = modelName
		self.early_stopping = EarlyStopping(patience = 5)
		if modelName == 'clf':
			self.model = classifier(n_class=6,hyp = hypModel)
			self.loss = torch.nn.CrossEntropyLoss()
		elif modelName == 'AE':
			self.model = ConvAutoencoder(hypModel)
			self.loss = torch.nn.MSELoss()

	# from torchsummary import summary
	# summary(self.model, (1, 50, 6))
	
	def setupTrain(self, trainSetup,datamodule):
		self.alpha = trainSetup['alpha']
		self.epochs = trainSetup['nEpochs']
		self.model.build()
		self.model = self.model.to(self.device).cuda()
		self.optimizer = optim.Adam(self.model.parameters(), lr=trainSetup['lr'])
		self.scheduler = StepLR(self.optimizer, trainSetup['step_size'], gamma=0.5)
		
		if trainSetup['penalty'] == 'mmd':
			self.penalty = MMDLoss()
		elif trainSetup['penalty'] == 'ot':
			self.penalty = OTLoss()
		elif trainSetup['penalty'] == 'ClDist':
			self.penalty = classDistance()
		self.name = self.name + '_' + trainSetup['penalty'] + '_'
		self.datamodule = datamodule

	def train(self, printGrad=False):
		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss
			train_loss = 0.0
			main_loss = 0.0
			penalty_loss = 0.0
			
			for i, batch in enumerate(self.datamodule.train_dataloader()):
				data, domain, label = batch['data'], batch['domain'], batch['label']
				data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
				                                                                         dtype=torch.int), label.to(
					self.device, dtype=torch.long)
				# self.model.train()
				self.optimizer.zero_grad()
				latent, pred = self.model(data)
				sourceIdx = np.where(domain.cpu() == 0)[0]
				sourceLabel = label[sourceIdx]
				pred = pred[sourceIdx]

				m_loss = self.loss(pred, sourceLabel)
				p_loss = self.penalty(latent, domain,sourceLabel)
				loss = m_loss + self.alpha * p_loss
				
				#loss = m_loss
				#loss.mean().backward()
				loss.backward()
				self.optimizer.step()
				train_loss += loss.mean().item()
				main_loss += m_loss.mean().item()
				penalty_loss += p_loss.mean().item()
			self.scheduler.step()
			train_loss = train_loss / i
			penalty_loss = penalty_loss / i
			main_loss = main_loss /i
			#print(next(self.model.parameters()).is_cuda)
			stops = self.validate(train_loss)
			histTrainLoss.append(train_loss)
			if stops:
				print(f'Early Stopped at epoch {epoch}')
				break
			# print(train_loss, '  ', main_loss, '  ', penalty_loss, '\n')
			if printGrad:
				self.getGrads()

		return histTrainLoss

	def validate(self,train_loss = None):
		valTarget, valSource, predValTarget, predValSource, val_loss = self.predict(stage = 'val')
		accValTarget = accuracy_score(valTarget, predValTarget)
		accValSource = accuracy_score(valSource, predValSource)
		self.early_stopping(val_loss)
		print('train_loss:  ',train_loss,'  | val_loss: ', val_loss, '|  acc Val -> |source: ', accValSource, '|   target: ', accValTarget)
		return self.early_stopping.early_stop

	def predict(self,stage = 'val',metrics = False):
		if stage is 'val':
			dataloaderEval = self.datamodule.val_dataloader()
		elif stage is 'test':
			dataloaderEval = self.datamodule.test_dataloader()
		else:
			raise ValueError(f"Oops!  stage {stage} is not defined")
		with torch.no_grad():
			for i, batch in enumerate(dataloaderEval):
				#TODO so funciona para batch = tamanho
				valTarget, valSource, predValTarget, predValSource,loss =  self._shared_eval_step(batch)
		if metrics:
			accTestTarget = accuracy_score(valTarget, predValTarget)
			accTestSource = accuracy_score(valSource, predValSource)
			outcomes = {}
			outcomes['accTestSource'] = accTestSource
			outcomes['accTestTarget'] = accTestTarget
			outcomes[stage + '_loss'] = loss
			return outcomes
			
			#print('acc Test -> |source: ', accTestSource, '|   target: ', accTestTarget)
		else:
			return valTarget, valSource, predValTarget, predValSource,loss
	
	def _shared_eval_step(self, batch):

		data, domain, label = batch['data'], batch['domain'], batch['label']
		#we can put the data in GPU to process but with 'no_grad' pytorch way?
		data, domain, label = data.to(self.device, dtype=torch.float), domain.to(self.device,
		                                                                         dtype=torch.int), label.to(
			self.device, dtype=torch.long)
		
		latent, pred = self.model(data)


		sourceIdx = np.where(domain.cpu() == 0)[0]
		targetIdx = np.where(domain.cpu() != 0)[0]
		
		sourceLabel = label[sourceIdx]
		y_hatSource = pred[sourceIdx]
		
		m_loss = self.loss(y_hatSource, sourceLabel)
		p_loss = self.penalty(latent, domain, sourceLabel)
		loss = m_loss + self.alpha * p_loss

		original = label.cpu().numpy()
		pred = np.argmax(pred.cpu().numpy(), axis=1)
		
		return original[targetIdx], original[sourceIdx], pred[targetIdx], pred[sourceIdx], loss.item()
	
	def save(self, savePath):
		with open(savePath, 'w') as s:
			pickle.dump(self.model, s, protocol=pickle.HIGHEST_PROTOCOL)
	
	def loadModel(self, filePath):
		with open(filePath, 'rb') as m:
			self.model = pickle.load(m)
	
	def getGrads(self):
		for name, param in self.model.named_parameters():
			print(name, param.grad.mean().item())
		print('\n\n')
		
