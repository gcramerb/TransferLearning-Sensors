import numpy as np
import glob,os
from sklearn.neighbors import KernelDensity



def simplest_SLselec(probs,trh):
	"""
	Select the samples with the highest probability class are higher than a trashhold
	:param probs:
	:param trh:
	:return:
	"""
	idx = np.where(probs.max(axis=1) > trh)[0]
	softLab = np.argmax(probs, axis=1)
	return idx,softLab

def saveSLdim(path_file,data, probs,first_save,trh = 0.75):
	"""
	Filter the ps labels and save in a npz file. Can be saved incrementaly or just save
	Save just the labels that were not saved before.
	
	:param path_file: the path to the pseudo Label file
	:param data: the data (X) that will be saved after the filtering
	:param probs: the classification probability of each sample prediction
	:param first_save: (bool) if is true, the data will be replaced anyway
	:param trh: the threshhold for filtering.
	:return:
	"""
	idx,SLab = simplest_SLselec(probs,trh)
	if data.shape[1] == 2:
		data = np.concatenate([data[:, [0], :,:], data[:, [1], :,:]], axis=-1)
	if os.path.isfile(path_file) and not first_save:
		with np.load(path_file, allow_pickle=True) as tmp:
			Xsl = tmp['X'].astype('float32')
			ysl = tmp['y']
			old_idx = tmp['new_idx']
		new_idx = np.array(list(set(idx) - set(old_idx)))
		if new_idx.size > 0:
			data = data[new_idx]
			SLab = SLab[new_idx]
			SLab = np.concatenate([SLab, ysl], axis=0)
			data = np.concatenate([data, Xsl], axis=0)
			idx = np.concatenate([new_idx,old_idx],axis = 0)
		else:
			SLab = ysl
			idx = old_idx
			data = Xsl
	else:
		new_idx = idx
		data = data[idx]
		SLab = SLab[idx]
	with open(path_file, "wb") as f:
		np.savez(f,X =data,y = SLab,folds = np.zeros(1),idx = idx)
	return new_idx


def saveSL(path_file, data, probs, first_save, trh=0.75):
	"""
	Filter the ps labels and save in a npz file. Can be saved incrementaly or just save

	:param path_file: the path to the pseudo Label file
	:param data: the data (X) that will be saved after the filtering
	:param probs: the classification probability of each sample prediction
	:param first_save: (bool) if is true, the data will be replaced anyway
	:param trh: the threshhold for filtering.
	:return:
	"""
	idx, SLab = simplest_SLselec(probs, trh)
	data = data[idx]
	SLab = SLab[idx]
	
	if data.shape[1] == 2:
		data = np.concatenate([data[:, [0], :, :], data[:, [1], :, :]], axis=-1)
	if os.path.isfile(path_file) and not first_save:
		with np.load(path_file, allow_pickle=True) as tmp:
			Xsl = tmp['X'].astype('float32')
			ysl = tmp['y']
		SLab = np.concatenate([SLab, ysl], axis=0)
		data = np.concatenate([data, Xsl], axis=0)
	with open(path_file, "wb") as f:
		np.savez(f, X=data, y=SLab, folds=np.zeros(1))
	return idx


def saveSL_gmm(path_file,data, probs,first_save,latent = None, trh = 0.75):
	"""
	The same method described before, but use a gaussian mixture model to select the pseudo label
	
	:param path_file: the path to the pseudo Label file
	:param data: the data (X) that will be saved after the filtering
	:param probs: the classification probability of each sample prediction
	:param first_save: (bool) if is true, the data will be replaced anyway
	:param trh: the threshhold for filtering.
	:return:
	"""
	idx,SLab = simplest_SLselec(probs,trh)
	print("old pl len: ", len(idx))
	new_idx = expandGMM(latent, probs)
	print(" new: ",len(new_idx))
	idx = list(set(idx).union(set(new_idx)))

	data= data[idx]
	SLab = SLab[idx]
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
	return idx


def expandGMM(data,probs):
	from sklearn.mixture import GaussianMixture
	"""
	proximo passo: testar a qualidade de cada cluster...
	eu estou 'expandindo com seguranca' pois escolho os que prob maior que 0.5 no pseudo lebel
	mas tambem os que estao no centro do cluster.
	"""
	gm = GaussianMixture(n_components=4, random_state=0).fit(data)
	GMMprobs = gm.predict_proba(data) # (n_samples, n_components)
	GMMpl = np.argmax(GMMprobs, axis=1)
	
	GMMidx = np.where(GMMprobs.max(axis=1) > 0.5)[0]
	idx_aux = np.where(probs.max(axis=1) > 0.5)[0]
	new_idx = list(set(idx_aux).intersection(set(GMMidx)))
	return new_idx


def simpleKernelProcess(latent,probs):
	"""
	Model some distribution based on the initial prob of each point.
	Calculate over each of the prob shape, to do not merge different possible classes

	O modelo Gussiano eh muito Flat...

	:param path_file:
	:param latentData:
	:return:
	"""

	dens_score = np.zeros_like(latent)
	for i in range(probs.shape[1]):
		kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(latent,sample_weight = probs[:,i])
		dens_score[:,i] = kde.score_samples(latent)
		del kde
	softLab = np.argmax(dens_score, axis=1)
	return softLab


def saveSL_kernel(path_file, data, probs, latent, first_save, trh=0.45):
	"""
	:param path_file: the path to the pseudo Label file
	:param data: the data (X) that will be saved after the filtering
	:param probs: the classification probability of each sample prediction
	:param first_save: (bool) if is true, the data will be replaced anyway
	:param latent
	:param trh: the threshhold for filtering.
	:return:
	"""
	
	idx, _ = simplest_SLselec(probs, trh)
	
	data = data[idx]
	latent = latent[idx]
	probs = probs[idx]
	
	SLab = simpleKernelProcess(latent,probs)

	if data.shape[1] == 2:
		data = np.concatenate([data[:, [0], :, :], data[:, [1], :, :]], axis=-1)
	if os.path.isfile(path_file) and not first_save:
		with np.load(path_file, allow_pickle=True) as tmp:
			Xsl = tmp['X'].astype('float32')
			ysl = tmp['y']
		SLab = np.concatenate([SLab, ysl], axis=0)
		data = np.concatenate([data, Xsl], axis=0)
	with open(path_file, "wb") as f:
		np.savez(f, X=data, y=SLab, folds=np.zeros(1))
	return idx
