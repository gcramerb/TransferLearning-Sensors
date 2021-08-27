import os
import numpy as np
from Utils.data import categorical_to_int
def processTorch(data,y,domain):
	dataP = []
	for i in range(len(data)):
		dataP.append((data[i],y[i],domain))
	return dataP


def getData(sourceDat,targetDat,inPath):
	file = os.path.join(inPath, f'{sourceDat}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	X = np.expand_dims(X, axis=1)
	y = categorical_to_int(y)
	source = X[:,:,:,0:6]
	source = processTorch(source,y,0)
	
	file = os.path.join(inPath, f'{targetDat}_f25_t2.npz')
	with np.load(file, allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
	X = np.expand_dims(X, axis=1)
	y = categorical_to_int(y)
	target= X[:,:,:,0:6]
	target = processTorch(target,y,1)
	return source+target

inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
train = getData('Dsads','Ucihar',inPath)


# def summarizeMetric(resList):
# 	resp = dict()
# 	mse = resList['MICE_f1']
# 	icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse), scale=st.sem(mse))
# 	mse = np.mean(mse)
# 	resp['MICE_f1'] = mse
# 	resp['MICE_f1_up'] = icMse[1]
# 	resp['MICE_f1_down'] = icMse[0]
#
# 	mse = resList['MF_f1']
# 	icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse), scale=st.sem(mse))
# 	mse = np.mean(mse)
# 	resp['MF_f1'] = mse
# 	resp['MF_f1_up'] = icMse[1]
# 	resp['MF_f1_down'] = icMse[0]
#
# 	mse = resList['EM_f1']
# 	icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse), scale=st.sem(mse))
# 	mse = np.mean(mse)
# 	resp['EM_f1'] = mse
# 	resp['EM_f1_up'] = icMse[1]
# 	resp['EM_f1_down'] = icMse[0]
#
# 	return resp