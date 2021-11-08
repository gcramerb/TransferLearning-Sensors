"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""
class TLmodel(LightningModule):
	
	def __init__(
			self,
			lr: float = 0.002,
			batch_size: list = [64,256],
			n_classes: int = 6,
			alpha: float = 1.0,
			penalty: str = 'mmd',
			data_shape: tuple = (1,50,6)
			clfHyp: dict = None,
			AEHyp: dict = None,
			**kwargs
	):
        super().__init__()
        self.save_hyperparameters()

        # networks
		self.clf = classifier(6, self.hparams.clfHyp)
		self.AE = ConvAutoencoder(self.hparams.AEHyp)
		self.clf.build()
		self.AE.build()
		
		#SET THE losses:
		self.recLoss = torch.nn.MSELoss()
		torch.nn.CrossEntropyLoss()
		self.clfLoss =
		if 'penalty' == 'mmd':
			self.discLoss = MMDLoss()
		elif 'penalty' == 'ot':
			self.discLoss = OTLoss()
		
		self.clDist = classDistance()
		self.optNames = ['Classifier','Reconstructior']
		

    def forward(self, X):
        return self.clf(X)

	def _shared_eval_step(self, batch,optimizer_idx,stage ='train'):
		source, target = batch
		dataSource, labSource = source['data'], source['label']
		dataTarget, labTarget = target['data'], target['label']
		
		# we can put the data in GPU to process but with 'no_grad' pytorch way?
		data, label = data.to(self.device, dtype=torch.float), label.to(self.device, dtype=torch.long)
		if optimizer_idx ==0:
			latent, pred = self(dataSource) #call forward method
			m_loss = self.clfLoss(pred, label)
			p_loss = self.clDist(latent, label)
			loss = m_loss + self.alpha * p_loss
		elif optimizer_idx ==1:
			latentT, decoded = AE.forward(dataTarget)
			m_loss = self.recLoss(dataTarget, decoded)
			latentS,pred = self.clf(dataSource)
			p_loss = self.discLoss(latentT,latentS)
			loss = m_loss + self.alpha * p_loss
		if stage =='val' or stage=='test':
			_, predTarget = self(dataTarget)
			return predSource, labSource, predTarget, labTarget, loss
			accSource = accuracy_score(labSource.cpu().numpy(), np.argmax(predSource.cpu().numpy(), axis=1))
			accTarget = accuracy_score(labTarget.cpu().numpy(), np.argmax(predTarget.cpu().numpy(), axis=1))
			loss = loss.item()
			
			metrics = {f"loss_{self.optNames[optimizer_idx]}": loss,
			           'accSource': accSource, 'accTarget': accTarget}
			return metrics
		return loss
			
    def training_step(self, batch, batch_idx, optimizer_idx):
		loss = _shared_eval_step(batch=batch,optimizer_idx=optimizer_idx)
		tqdm_dict = {f"{self.optNames[optimizer_idx]}_loss": loss}
		output = OrderedDict({f"{self.optNames[optimizer_idx]}_loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output

	def set_requires_grad(model, requires_grad=True):
		for param in self.clf.parameters():
			param.requires_grad = requires_grad
		for param in self.AE.parameters():
			param.requires_grad = requires_grad
			
	def configure_optimizers(self):
		lr = self.hparams.lr
		opt_clf = torch.optim.Adam(self.clf.parameters(), lr=lr)
		opt_AE = torch.optim.Adam(self.AE.parameters(), lr=lr)
		return [opt_clf, opt_AE], []

	def validation_step(self, batch, batch_idx):
		with torch.no_grad():
			metrics = self._shared_eval_step(batch, batch_idx, stage = 'val')
			# self.logger.experiment.log_dict('1',metrics,'val_metrics.txt')
			self.log('val_loss', metrics['loss1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
			self.log('accValSource', metrics['accSource'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
			self.log('accValTarget', metrics['accTarget'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
			return metrics