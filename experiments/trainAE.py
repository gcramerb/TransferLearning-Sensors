
from autoencoder import ConvAutoencoder
from geomloss import SamplesLoss

class AEnetwork:
	def __init__(self, bs=32):
		self.bs = bs
	
	def buildModel(self, hyp=None):
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.model = ConvAutoencoder(hyp).to(self.device)
	def setTrainOpt(self):
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		self.scheduler = StepLR(optimizer, step_size=30, gamma=0.4)
		# criterion = SoftDTW(use_cuda, gamma=0.01)
		self.criterion = nn.MSELoss()
		self.epochs = 70

	def train(self, xTrain):
		trainloader = DataLoader(xTrain, shuffle=True, batch_size=self.bs)
		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(self.epochs):
			# monitor training loss
			train_loss = 0.0
			for (i,data) in trainloader:

				acc,gyr,domain = data
				acc, gyr, domain = acc.to(self.device,dtype=torch.float),gyr.to(self.device,dtype=torch.float),domain.to(self.device, dtype=torch.int)
				optimizer.zero_grad()
				latent,rec = self.model([acc,gyr])
				
				sensor = torch.cat(tuple(acc,gyr))
				loss = criterion(latent,domain, rec, sensor)
				loss.mean().backward()
				optimizer.step()
				train_loss += loss.mean().item()
			
			scheduler.step()
			train_loss = train_loss / len(trainloader)
			histTrainLoss.append(train_loss)
		return histTrainLoss

	def predict(self, xTest):
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