



def processTorch(acc,gyr,domain):
	data = []
	for i in len(acc):
		data.append((acc[i],gyr[i],domain[i]))
	return data


def getDataTorch(self,sourceDat,targetDat,inPath):
	file = os.path.join(inPath, f'{sourceDat}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	X = np.expand_dims(X, axis=1)
	sourceAcc,sourceGyr = X[:,:,:,0:3],X[:,:,:,3:6]
	source = processTorch(sourceAcc,sourceGyr,y,0)
	
	file = os.path.join(inPath, f'{targetDat}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	X = np.expand_dims(X, axis=1)
	targetAcc,targetGyr = X[:,:,:,0:3],X[:,:,:,3:6]
	target = processTorch((targetAcc,targetGyr,y,1))
	return source,target





