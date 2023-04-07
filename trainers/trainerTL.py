import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse,glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')

from models.classifier import classifier
from models.blocks import Encoder, discriminator, domainClf
from models.customLosses import MMDLoss, OTLoss, CenterLoss, SinkhornDistance, CORAL

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that trainers basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""
#TODO: implementar que cada modelo treine mais de uma epoca..

class TLmodel(LightningModule):
	
	def __init__(
			self,
			trainParams: dict = None,
			useMixup: bool =True,
			lossParams: dict = None,
			save_path: str = None,
			class_weight: torch.tensor = None,
			n_classes = 4,
			**kwargs
	):
		super().__init__()
		self.hparams.alpha = trainParams['alpha']
		self.hparams.beta = trainParams['beta']
		self.hparams.penalty = trainParams['discrepancy']
		self.hparams.weight_decay = trainParams['weight_decay']
		self.hparams.dropout_rate = trainParams['dropout_rate']
		self.hparams.lr_fe = trainParams['lr']
		self.hparams.lr_disc = trainParams['lr']
		self.hparams.input_shape = trainParams['input_shape']
		self.hparams.n_classes = n_classes
		self.save_hyperparameters()
		
	def create_model(self):

		self.FE = Encoder(hyp=self.hparams.trainParams,
			              input_shape=self.hparams.input_shape)
		self.FE.build()
		
		self.Disc  =  discriminator(dropout_rate = self.hparams.dropout_rate,
		                                 encoded_dim = self.hparams.trainParams['enc_dim'],
		                                 n_classes = self.hparams.n_classes)
		self.Disc.build()
		
		self.clfLoss = nn.CrossEntropyLoss(weight=self.hparams.class_weight)
		
		self.classDist = CenterLoss( num_classes=self.hparams.n_classes, feat_dim=self.hparams.trainParams['enc_dim'], use_gpu=True)

		if self.hparams.penalty == 'mmd':
			self.discLoss = MMDLoss()
		elif self.hparams.penalty == 'ot':
			#self.discLoss = OTLoss(hyp=lossParams)
			self.discLoss = OTLoss()
		elif self.hparams.penalty == 'skn':
			self.discLoss = SinkhornDistance(eps=1e-3, max_iter=200)
		elif self.hparams.penalty == 'coral':
			self.discLoss = CORAL()
		else:
			raise ValueError('specify a valid discrepancy loss!')

	def load_params(self, save_path, file):
		PATH = os.path.join(save_path, file + '_feature_extractor')
		self.FE.load_state_dict(torch.load(PATH))
		PATH = os.path.join(save_path, file + '_discriminator')
		self.Disc.load_state_dict(torch.load(PATH))
		for param in self.FE.parameters():
			param.requires_grad = True
		for param in self.Disc.parameters():
			param.requires_grad = True

	def save_params(self,save_path,file):
		path = os.path.join(save_path,file + '_feature_extractor')
		torch.save(self.FE.state_dict(), path)
		path = os.path.join(save_path,file + '_discriminator')
		torch.save(self.Disc.state_dict(), path)
	
	def mixup(self, latent, label, weight):
		indices = torch.randperm(latent.size(0), device=latent.device, dtype=torch.long)
		perm_latent = latent[indices]
		perm_label = label[indices]
		return latent.mul(weight).add(perm_latent, alpha=1 - weight), label.mul(weight).add(perm_label,
		                                                                                    alpha=1 - weight)
	
	def compute_loss(self, batch,optimizer_idx):
		logs = {}
		source, target = batch[0], batch[1]
		dataSource, labSource = source['data'], source['label'].long()
		
		latentS= self.FE(dataSource)
		if(self.hparams.useMixup):
			latentSMixup, labelSMixup = self.mixup(latentS, labSource, np.random.beta(0.5, 0.5))
			# latentSMixup = torch.cat((latentS, latentSMixup), 0)
			# labelSMixup = torch.cat((labSource, labelSMixup), 0)
			classDistence = self.classDist(latentS, labSource.argmax(axis=1))
		else:
			latentSMixup = latentS
			labelSMixup = labSource
			classDistence = self.classDist(latentS, labSource)

		predS = self.Disc(latentSMixup)
		
		loss = self.clfLoss(predS, labelSMixup) + self.hparams.beta * classDistence
		logs['clf loss'] = loss.detach()
		logs['classDistence'] = classDistence.detach()
		if optimizer_idx ==0: # updating FE
			dataTarget = target['data']
			latentT = self.FE(dataTarget)
			discrepancy = self.discLoss(latentT, latentS)
			loss = loss + self.hparams.alpha * discrepancy
			logs['discrepancy'] = discrepancy.detach()

		logs['loss'] = loss.detach()

		return loss,logs
	
	# def _shared_eval_step(self, batch,optimizer_idx,stage='val'):
	#
	#
	# 	if stage == 'val' and optimizer_idx is not None:
	# 		_, logs = self.compute_loss(batch,optimizer_idx)
	# 		for k, v in logs.items():
	# 			metrics[stage + '_' + k] = v.detach()
	# 	return metrics
	
	def training_step(self, batch, batch_idx,optimizer_idx):
		loss, log = self.compute_loss(batch,optimizer_idx)
		tqdm_dict = log
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		self.log("training_loss", loss,batch_size=self.batch_size)
		return output
	
	def training_epoch_end(self, output):
		metrics = {}
		if isinstance(output[0], list):
			opt = [i['log'] for i in output[0]]
		else:
			opt = [i['log'] for i in output]

		metrics['clf loss']= []
		metrics['loss'] = []
		metrics['discrepancy'] = []
		metrics['classDistence'] = []
		for i in opt:
			for k in i.keys():
				metrics[k].append(i[k].cpu().data.numpy().item())
		for k, v in metrics.items():
			self.log(k, np.mean(v), on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size = self.batch_size)
		return None
	
	def validation_step(self, batch, batch_idx):
		return batch
	
	def validation_epoch_end(self, out):
		YsourcePred = []
		YtargetPred = []
		YsourceTrue = []
		YtargetTrue = []
		for batch in out:
			source, target = batch[0], batch[1]
			dataSource, labSource = source['data'], source['label'].long()
			dataTarget, labTarget = target['data'], target['label'].long()
			latS = self.FE(dataSource)
			pred_S = self.Disc(latS)
			latT = self.FE(dataTarget)
			pred_T = self.Disc(latT)
			YsourcePred.append(np.argmax(pred_S.detach().cpu().numpy(), axis=1))
			YtargetPred.append(np.argmax(pred_T.detach().cpu().numpy(), axis=1))
			YsourceTrue.append(labSource.cpu().numpy())
			YtargetTrue.append(labTarget.cpu().numpy())
		
		YsourceTrue = np.concatenate(YsourceTrue,axis =0)
		YtargetTrue = np.concatenate(YtargetTrue, axis=0)
		YsourcePred = np.concatenate(YsourcePred, axis=0)
		YtargetPred = np.concatenate(YtargetPred, axis=0)
		metrics = {}
		if(len(YsourceTrue.shape)>1):
			metrics['valAccSource'] = accuracy_score(np.argmax(YsourceTrue, axis=1),YsourcePred)
			metrics['valAccTarget'] = accuracy_score(np.argmax(YtargetTrue, axis=1),YtargetPred)
		else:
			metrics['valAccSource'] = accuracy_score(YsourceTrue,YsourcePred)
			metrics['valAccTarget'] = accuracy_score(YtargetTrue,YtargetPred)
			

		# return metrics
		# keys_ = out[0].keys()
		# for k in keys_:
		# 	val = [i[k] for i in out]
		# 	if 'acc' in k:
		# 		metrics[k] = np.mean(val)
		# 	else:
		# 		metrics[k] = torch.mean(torch.stack(val))
		
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size = self.batch_size)
		return None
	
	def configure_optimizers(self):
		opt_list = []
		opt_list.append(torch.optim.RMSprop(self.FE.parameters(),
		                                 lr=self.hparams.lr_fe,
		                                 weight_decay=self.hparams.weight_decay))
		
		opt_list.append(torch.optim.Adam(self.Disc.parameters(), lr=self.hparams.lr_disc))
		
		# lr_sch = StepLR(opt_FE, step_size=20, gamma=0.5)
		return opt_list, []
	
	def getPredict(self,domain = 'Target'):
		"""
		:return:
		"""
		
		with torch.no_grad():
			if domain =='Source':
				out = self._predict(self.dm_source.test_dataloader(), 'Source')
			elif domain =='Target':
				out = self._predict(self.dm_target.test_dataloader(), 'Target')
			else:
				raise ValueError('You must specify a valid domain!')
		return out
	def get_final_metrics(self):
		result = {}
		pred_source = self.getPredict(domain = 'Source')
		pred_target = self.getPredict(domain='Target')
		predictions = {**pred_source,**pred_target}

		result['acc_source_all'] = accuracy_score(predictions['trueSource'], predictions['predSource'])
		result['acc_target_all'] = accuracy_score(predictions['trueTarget'], predictions['predTarget'])
		result['cm_source'] = confusion_matrix(predictions['trueSource'], predictions['predSource'])
		result['cm_target'] = confusion_matrix(predictions['trueTarget'], predictions['predTarget'])
		return result
	

	def _predict(self, dataloader,domain):
		with torch.no_grad():
			latent = []
			probs = []
			y_hat = []
			true = []
			data_ori = []
			for data in dataloader:
				X, y = data['data'], data['label'].long()
				lat = self.FE(X)
				p = self.Disc(lat).cpu().numpy()
				y_hat.append(np.argmax(p, axis=1))
				latent.append(lat.cpu().numpy())
				probs.append(p)
				true.append(y.cpu().numpy())
				data_ori.append(X)
			predictions = {}
			predictions['latent' +domain ] = np.concatenate(latent, axis=0)
			predictions['pred' + domain] = np.concatenate(y_hat, axis=0)
			predictions['true' + domain] = np.concatenate(true, axis=0)
			if(len(predictions['true' + domain].shape)>1):
				predictions['true' + domain] = np.argmax(predictions['true' + domain],axis =1)
			predictions['prob'  + domain] = np.concatenate(probs, axis=0)
			predictions['data' + domain] = np.concatenate(data_ori, axis=0)
			return predictions
	
	def train_dataloader(self):
		loaders = [self.dm_source.train_dataloader(),
		        self.dm_target.train_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders

	def test_dataloader(self):
		loaders = [self.dm_source.test_dataloader(),
		           self.dm_target.test_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders

	def val_dataloader(self):
		loaders = [self.dm_source.val_dataloader(),
		           self.dm_target.val_dataloader()]
		combined_loaders = CombinedLoader(loaders, "max_size_cycle")
		return combined_loaders
	
	def setDatasets(self, dm_source = None, dm_target = None):
		if dm_source:
			self.dm_source = dm_source
			self.n_classes = dm_source.n_classes
			self.batch_size = dm_source.batch_size
		if dm_target:
			self.dm_target = dm_target
			self.n_classes = dm_target.n_classes
			self.batch_size = dm_target.batch_size
		# if self.batch_size != dm_source.batch_size:
		# 	raise ValueError("Differents Batch size!")

