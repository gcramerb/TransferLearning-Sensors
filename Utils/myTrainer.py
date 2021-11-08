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

import sys, os, argparse, pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier,classifierTest
from models.autoencoder import ConvAutoencoder
from models.customLosses import MMDLoss, OTLoss, classDistance
# import geomloss



from dataProcessing.create_dataset import crossDataset, targetDataset, getData
from dataProcessing.dataModule import SingleDatasetModule

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
	def __init__(self):
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")


	def setupTrain(self, modelName,setup,source,target,hypModel):
		"""
		:param setup: (dict): all training configurations
		:param source: (datamodule)
		:param target: (datamodule)
		:return: None
		"""
		if source.dataFormat == 'Matrix':
			self.clf = classifier(n_class=6,modelName=modelName, hyp=hypModel,inputShape=(1,50,6))
		elif source.dataFormat == "Channel":
			self.clf = classifier(n_class=6,modelName=modelName, hyp=hypModel, inputShape=(2,50,3))
		else:
			raise ValueError("put a valid DataFormat" )
		
		self.loss = torch.nn.CrossEntropyLoss()
		self.early_stopping = EarlyStopping(patience = 10)
		self.valLoss = []
		
		self.alpha = setup['alpha']
		self.epochs = setup['nEpochs']
		self.clf.build()
		self.clf = self.clf.to(self.device).cuda()
		self.optimizer = optim.Adam(self.clf.parameters(), lr=setup['lr'])
		self.scheduler = StepLR(self.optimizer, setup['step_size'], gamma=0.5)
		
		if setup['penalty'] == 'mmd':
			self.penalty = MMDLoss()
		elif setup['penalty'] == 'ot':
			self.penalty = OTLoss()
		elif setup['penalty'] == 'ClDist':
			self.penalty = classDistance()

		self.dm_source = source
		self.dm_target = target

	def train(self, printGrad=False):
		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss1
			train_loss = 0.0
			main_loss = 0.0
			penalty_loss = 0.0
			
			for i, batch in enumerate(self.dm_source.train_dataloader()):
				self.optimizer.zero_grad()
				label,pred,loss = self._shared_eval_step(batch)
				#loss1.mean().backward()
				loss.backward()
				self.optimizer.step()
				train_loss += loss.mean().item()
				# main_loss += m_loss.mean().item()
				# penalty_loss += p_loss.mean().item()
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
			#print(train_loss, '  ', main_loss, '  ', penalty_loss, '\n')
			if printGrad:
				self.getGrads()

		return histTrainLoss

	def validate(self,train_loss = None):
		labelValSource, labelValTarget, predValSource,predValTarget, val_loss = self.predict(stage = 'val')
		accValSource = accuracy_score(labelValSource, predValSource)
		accValTarget = accuracy_score(labelValTarget, predValTarget)

		self.early_stopping(val_loss)
		self.valLoss.append(val_loss)
		print('train_loss:  ',train_loss,'  | val_loss: ', val_loss, '|  acc Val -> |source: ', accValSource, '|   target: ', accValTarget)
		return self.early_stopping.early_stop

	def predict(self,stage = 'val',metrics = False):
		#use all target data to evaluate?
		if stage is 'val':
			dataEval_source = self.dm_source.val_dataloader()
			dataEval_target = self.dm_target.val_dataloader()
		elif stage is 'test':
			dataEval_source = self.dm_source.test_dataloader()
			dataEval_target = self.dm_target.test_dataloader()
		else:
			raise ValueError(f"Oops!  stage {stage} is not defined")
		with torch.no_grad():
			for i, batch in enumerate(dataEval_source):
				labelSource, predSource,loss =  self._shared_eval_step(batch)
				predSource = np.argmax(predSource.cpu().numpy(), axis=1)
				labelSource = labelSource.cpu().numpy()
			for i, batch in enumerate(dataEval_target):

				labelTarget, predTarget,_ =  self._shared_eval_step(batch)
				predTarget = np.argmax(predTarget.cpu().numpy(), axis=1)
				labelTarget = labelTarget.cpu().numpy()
				

		if metrics:
			accSource = accuracy_score(labelSource, predSource)
			accTarget = accuracy_score(labelTarget, predTarget)
			outcomes = {}
			outcomes['acc_' + stage+'_Source'] = accSource
			outcomes['acc_' + stage+'_Target'] = accTarget
			outcomes[stage + '_loss'] = loss.item()
			return outcomes

		else:
			return labelSource, labelTarget,predSource,predTarget,loss.item()
	
	def _shared_eval_step(self, batch):

		data,  label = batch['data'], batch['label']
		#we can put the data in GPU to process but with 'no_grad' pytorch way?
		data, label = data.to(self.device, dtype=torch.float), label.to(self.device, dtype=torch.long)
		latent, pred = self.clf(data)
		m_loss = self.loss(pred, label)
		p_loss = self.penalty(latent, label)
		loss = m_loss + self.alpha * p_loss


		return label,pred, loss
	
	def save(self, savePath):
		with open(savePath, 'wb') as s:
			pickle.dump(self.clf, s, protocol=pickle.HIGHEST_PROTOCOL)
	
	def loadModel(self, filePath):
		with open(filePath, 'rb') as m:
			self.clf = pickle.load(m)
	
	def getGrads(self):
		for name, param in self.model.named_parameters():
			print(name, param.grad.mean().item())
		print('\n')
		
