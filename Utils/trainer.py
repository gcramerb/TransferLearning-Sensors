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
from models.customLosses import MMDLoss,OTLoss,classDistance
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
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict
class networkLight(LightningModule):
	def __init__(
			self,
			latent_dim: int = 50,
			lr: float = 0.0002,
			batch_size: int = 128,
			n_classes: int = 6,
			alpha: float = 1.2,
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
			return torch.nn.CrossEntropyLoss(), OTLoss()
		elif penalty == 'clDist':
			return torch.nn.CrossEntropyLoss(), classDistance()
	
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
		loss = m_loss(pred, true) + self.hparams.alpha * p_loss(latent, domain,label)
		
		tqdm_dict = {"loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	def training_step_end(self , training_step_outputs):
		metrics = training_step_outputs['log']
		loss = metrics['loss'].cpu().data.numpy().item()
		# self.logger.experiment.log_metric(self.logger.run_id, key='training_loss',
		#                                   value=loss)
		self.log('training_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	#print('training_loss:  ',loss)
	def validation_step(self, batch, batch_idx):

		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']

		#self.logger.experiment.log_dict('1',metrics,'val_metrics.txt')
		self.log('val_loss', metrics['loss'], on_step=False, on_epoch=True, prog_bar=True,logger = True)
		self.log('accValSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True,logger = True)
		self.log('accValTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True,logger = True)
		
		# self.logger.experiment.log_metric(self.logger.run_id,key= 'val_loss', value= metrics['loss'])
		# self.logger.experiment.log_metric(self.logger.run_id,'accValSource', metrics['accSource'])
		# self.logger.experiment.log_metric(self.logger.run_id,'accValTarget', metrics['accTarget'])
		#print('val_loss: ',  metrics['loss'],' ','accValSource: ',
		 #                                    metrics['accSource'],' ','accValTarget: ',metrics['accTarget'])
		return metrics
	
	def test_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		# self.log_dict(metrics)
		# self.logger.experiment.log_metric(self.logger.run_id,'accTestSource', metrics['accSource'])
		# self.logger.experiment.log_metric(self.logger.run_id,'accTestTarget', metrics['accTarget'])
		self.log('accTestSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True,logger = True)
		self.log('accTestTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True,logger = True)
		return metrics
	
	def _shared_eval_step(self, batch, batch_idx):
		data, domain, label = batch['data'], batch['domain'], batch['label']
		latent, pred = self.clf(data)
		
		sourceIdx = np.where(domain.cpu() == 0)[0]
		targetIdx = np.where(domain.cpu() != 0)[0]
		trueSource = label[sourceIdx]
		predSource = pred[sourceIdx]
		trueTarget = label[targetIdx]
		predTarget = pred[targetIdx]
		outputs = (trueSource, predSource, trueTarget, predTarget)
		# return outputs
		trueSource, predSource, trueTarget, predTarget = outputs
		m_loss, p_loss = self.myLoss(self.hparams.penalty)
		trueSource = trueSource.long()  # why need this?
		loss = m_loss(predSource, trueSource)  + self.hparams.alpha * p_loss(latent,domain,label)
		accSource = accuracy_score(trueSource.cpu().data.numpy(), np.argmax(predSource.cpu().data.numpy(), axis=1))
		accTarget = accuracy_score(trueTarget.cpu().data.numpy(), np.argmax(predTarget.cpu().data.numpy(), axis=1))
		loss = loss.cpu().data.numpy().item()
		
		metrics = {"loss": loss,
		           'accSource': accSource, 'accTarget': accTarget}
		
		tqdm_dict = metrics
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		}
		return result
	def predict(self,dataTest):
		latent,domain,pred = [],[],[]
		data = np.concatenate([x['data'] for x in dataTest])
		d = [x['domain'] for x in dataTest]
		label = [x['label'] for x in dataTest]

		l, p = self.clf.forward(torch.tensor(np.expand_dims(data,axis = 1)))
		pred = np.argmax(p.data.numpy(),axis = 1)
		return l.data.numpy(),d,pred.tolist(),label
	
	def configure_optimizers(self):
		opt = optim.Adam(self.clf.parameters(), lr=self.hparams.lr)
		scheduler = StepLR(opt, step_size=25, gamma=0.1)
		return {"optimizer":opt,"lr_scheduler" : scheduler}
	
	def on_epoch_end(self):
		pass
	

