import numpy as np

def generate_pseudoLab(path_file,trh):
	with np.load(path_file, allow_pickle=True) as tmp:
		Xsl = tmp['Xsl'].astype('float32')
		probs = tmp['ysl']
	idx = np.where(probs.max(axis = 0 ) >trh)[0]
	softL = np.argmax(probs[idx], axis=1)
	dataSL = Xsl[idx]
	return dataSL,softL

def dicrepPerClass():
	"""
	Not just analyse if the max prob is upper a certain thrashhold, but also
	check fi the dicrepance of that point is lower in the same class in source
	
	:return:
	"""

def kernelProcess(path_file,latentData):
	"""
	Model some distribution based on the initial prob of each point.
	maybe a mixture of gaussian (or student-T)
	Calculate over each of the prob shape, to do not merge different possible classes
	
	
	:param path_file:
	:param latentData:
	:return:
	"""
	