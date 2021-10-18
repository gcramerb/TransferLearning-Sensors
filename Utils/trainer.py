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


class myTrainer:
	def __init__(self, hyp=None):
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.name = hyp['model'] + '_' + hyp['penalty'] + 'network'
		if hyp['model'] == 'clf':
			self.model = classifier(n_class=6)
			self.loss = torch.nn.CrossEntropyLoss()
		elif hyp['model'] =='AE':
			self.model = ConvAutoencoder(hyp['model_hyp'])
			self.loss = torch.nn.MSELoss()
		self.model.build()
		self.model = self.model.to(self.device).cuda()
		self.optimizer = optim.Adam(self.model.parameters(), lr=hyp['lr'])
		self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)

		if hyp['penalty'] == 'mmd':
			self.penalty = MMDLoss()
		elif hyp['penalty'] == 'ot':
			self.penalty = OTLoss()
	
	# from torchsummary import summary
	# summary(self.model, (1, 50, 6))
	
	def configTrain(self, alpha=1.8, n_ep=120, bs=128):
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
				loss =  m_loss + self.alpha * p_loss
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
				accValTarget = accuracy_score(valTarget,predValTarget)
				accValSource = accuracy_score(valSource,predValSource)
				
				print('acc Val -> |source: ',accValSource,'|   target: ',accValTarget )

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

from pytorch_lightning import LightningDataModule,LightningModule
from collections import OrderedDict
class networkLight(LightningModule):
	def __init__(
			self,
			latent_dim: int = 50,
			lr: float = 0.0002,
			batch_size: int = 128,
			n_classes: int = 6,
			alpha: float = 0.2,
			penalty: str = 'mmd',
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.clf = classifier(self.hparams.n_classes)
		self.clf.build()
	
	def myLoss(self, penalty):
		if penalty == 'mmd':
			return torch.nn.CrossEntropyLoss(), MMDLoss()
		elif penalty == "ot":
			torch.nn.CrossEntropyLoss(), OTLoss()
	
	def forward(self, X):
		return self.clf(X)
	
	# def adversarial_loss(self, y_hat, y):
	#     return F.binary_cross_entropy(y_hat, y)
	
	def training_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		latent, pred = self.clf(data)
		sourceIdx = np.where(domain.cpu() == 0)[0]
		true = label[sourceIdx]
		pred = pred[sourceIdx]
		m_loss, p_loss = self.myLoss(self.hparams.penalty)
		true = true.long()  # why need this?
		loss = self.hparams.alpha * m_loss(pred, true) + (1 - self.hparams.alpha) * p_loss(latent, domain)
		
		tqdm_dict = {"loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	#def on
	
	def validation_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		latent, pred = self.clf(data)
		
		sourceIdx = np.where(domain.cpu() == 0)[0]
		targetIdx = np.where(domain.cpu() != 0)[0]
		trueSource = label[sourceIdx]
		predSource = pred[sourceIdx]
		trueTarget = label[targetIdx]
		predTarget = pred[targetIdx]
		outputs = (trueSource,predSource,trueTarget,predTarget)
		return outputs
		#pred = np.argmax(pred.cpu().data.numpy(), axis=1)
	def validation_end(self, outputs):
		trueSource, predSource, trueTarget, predTarget = outputs
		m_loss, p_loss = self.myLoss(self.hparams.penalty)
		trueSource = trueSource.long()  # why need this?
		val_loss = self.hparams.alpha * m_loss(predSource, trueSource) + (1 - self.hparams.alpha) * p_loss(latent,
		                                                                                                   domain)
		accValSource = accuracy_score(trueSource.cpu().data.numpy(), np.argmax(predSource.cpu().data.numpy(), axis=1))
		accValTarget = accuracy_score(trueTarget.cpu().data.numpy(), np.argmax(predTarget.cpu().data.numpy(), axis=1))
		
		metrics = {"val_loss": val_loss.cpu().data.numpy().item(),
		           'val_acc_source': accValSource, 'val_acc_target': accValTarget}
		tqdm_dict = metrics
		
		# self.logger.experiment.log_dict('1',metrics,'val_metrics.txt')
		self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log('accValSource', accValSource, on_step=True, on_epoch=True, prog_bar=True)
		self.log('accValTarget', accValTarget, on_step=True, on_epoch=True, prog_bar=True)
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		}
		return result
		
	def configure_optimizers(self):
		return optim.Adam(self.clf.parameters(), lr=self.hparams.lr)
	
	def on_epoch_end(self):
		pass

