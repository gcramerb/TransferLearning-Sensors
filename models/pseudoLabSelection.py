import numpy as np
import glob,os
from sklearn.neighbors import KernelDensity



def simplest_SLselec(probs,trh):
	idx = np.where(probs.max(axis=1) > trh)[0]
	softLab = np.argmax(probs[idx], axis=1)
	return idx,softLab

def saveSL(path_file,data, probs,first_save,trh = 0.35):
	idx,SLab = simplest_SLselec(probs,trh)
	print(f'\n Len idx: {len(idx)} \n')
	data= data[idx]
	if data.shape[1] == 2:
		data = np.concatenate([data[:, [0], :,:], data[:, [1], :,:]], axis=-1)
	if os.path.isfile(path_file) and not first_save:
		with np.load(path_file, allow_pickle=True) as tmp:
			Xsl = tmp['X'].astype('float32')
			ysl = tmp['y']
		SLab = np.concatenate([SLab, ysl], axis=0)
		data = np.concatenate([data, Xsl], axis=0)
	with open(path_file, "wb") as f:
		np.savez(f,X =data,y = SLab,folds = np.zeros(1))
	return len(data)

def simpleKernelProcess(path_file,trh = 0.75):
	"""
	Model some distribution based on the initial prob of each point.
	Calculate over each of the prob shape, to do not merge different possible classes

	O modelo Gussiano eh muito Flat...

	:param path_file:
	:param latentData:
	:return:
	"""
	with np.load(path_file, allow_pickle=True) as tmp:
		X = tmp['dataSL'].astype('float32')
		latent = tmp['latentSL']
		probs = tmp['ySL']
	dens_score =  np.zeros_like(probs)
	for i in range(probs.shape[1]):
		kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(latent,sample_weight = probs[:,i])
		dens_score[:,i] = kde.score_samples(latent)
		del kde
	
	#entender o que eh o score e definir: como escolher.
	# nao faz muito sentido escolher o score relativo (percentil)
	idx = np.where(dens_score.max(axis=0) > trh)[0]
	softL = np.argmax(probs[idx], axis=1)
	dataSL = X[idx]
	return dataSL,softL


def kernelProcess(path_file):
	"""
	Model some distribution based on the initial prob of each point.
	maybe a mixture of gaussian (or student-T)
	Calculate over each of the prob shape, to do not merge different possible classes


	:param path_file:
	:param latentData:
	:return:
	"""
	with np.load(path_file, allow_pickle=True) as tmp:
		X = tmp['dataSL'].astype('float32')
		latent = tmp['latentSL']
		probs = tmp['ySL']



def dicrepPerClass(path_file,thr = 0.5):
	"""
	Not just analyse if the max prob is upper a certain thrashhold, but also
	check fi the dicrepance of that point is lower in the same class in source
	
	:return:
	"""
	with np.load(path_file, allow_pickle=True) as tmp:
		X = tmp['dataSL'].astype('float32')
		latent = tmp['latentSL']
		probs = tmp['ySL']
	

