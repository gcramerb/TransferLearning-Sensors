import numpy as np
import glob,os
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

def getPseudoLabel(prediction, method = 'simplest'):
	"""
	Filter the ps labels and return the indexes, the trueLabels and the Pseudo lables.

	:param prediction: a dictionary with all the important data: X, y (true), probabilitys, latent data.
	:param trh: the threshhold for filtering, just for simplest method
	:return:
	"""
	if method == 'simplest':
		idx = simplest(prediction['probs'], 0.95)
		softLabel = np.argmax(prediction['probs'], axis=1)
		softLabel = softLabel[idx]
		data = prediction['data'][idx]
		trueLabel =  prediction['true'][idx]
		
	if method =='gmm':
		data, softLabel, trueLabel = GMM(prediction, 0.75)

	if method =='kernel':
		softLabel = simpleKernelProcess(prediction)
		data = prediction['data']
		trueLabel = prediction['true']

	if data.shape[1] == 2:
		data = np.concatenate([data[:, [0], :, :], data[:, [1], :, :]], axis=-1)
	return data, softLabel,trueLabel




def simplest(probs, trh):
	
	"""
	Select the samples with the highest probability class are higher than a trashhold
	:param probs:
	:param trh:
	:return:
	"""
	idx = np.where(probs.max(axis=1) > trh)[0]
	return idx


def GMM(prediction, trh=0.75):
	"""
	The same method described before, but use a gaussian mixture model to select the pseudo label

	:param path_file: the path to the pseudo Label file
	:param data: the data (X) that will be saved after the filtering
	:param probs: the classification probability of each sample prediction
	:param first_save: (bool) if is true, the data will be replaced anyway
	:param trh: the threshhold for filtering.
	:return:
	"""
	idx = simplest(prediction['probs'], trh)
	print("old pl len: ", len(idx))
	gm = GaussianMixture(n_components=4, random_state=0).fit(prediction['latent'])
	GMMprobs = gm.predict_proba(prediction['latent'])  # (n_samples, n_components)
	GMMpl = np.argmax(GMMprobs, axis=1)
	
	GMMidx = np.where(GMMprobs.max(axis=1) > 0.5)[0]
	idx_aux = np.where(prediction['probs'].max(axis=1) > 0.5)[0]
	new_idx = list(set(idx_aux).intersection(set(GMMidx)))
	
	print(" new: ", len(new_idx))
	idx = list(set(idx).union(set(new_idx)))
	data = prediction['data'][idx]
	softLabel = np.argmax(prediction['probs'], axis=1)
	softLabel = softLabel[idx]
	return data, softLabel, prediction['true'][idx]


def simpleKernelProcess(prediction):
	"""
	Model some distribution based on the initial prob of each point.
	Calculate over each of the prob shape, to do not merge different possible classes

	O modelo Gussiano eh muito Flat...

	:param path_file:
	:param latentData:
	:return: Just the pseudo label because all the samples get a pseudo label association.
	"""
	dens_score = np.zeros_like(prediction['latent'])
	for i in range(prediction['probs'].shape[1]):
		kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(prediction['latent'], sample_weight=prediction['probs'][:, i])
		dens_score[:, i] = kde.score_samples(prediction['latent'])
		del kde
	softLab = np.argmax(dens_score, axis=1)
	return softLab





def saveAllLabels(path_file, data, trueLabel, softLabel, latent):
	if data.shape[1] == 2:
		data = np.concatenate([data[:, [0], :, :], data[:, [1], :, :]], axis=-1)
	# if os.path.isfile(path_file) and not first_save:
	# 	with np.load(path_file, allow_pickle=True) as tmp:
	# 		Xsl = tmp['X'].astype('float32')
	# 		ysl = tmp['y']
	# 	SLab = np.concatenate([SLab, ysl], axis=0)
	# 	data = np.concatenate([data, Xsl], axis=0)
	with open(path_file, "wb") as f:
		np.savez(f, X=data, y=trueLabel, pseudoLabel =softLabel,  folds=np.zeros(1))
	return True






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
	idx,SLab = simple(probs,trh)
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

