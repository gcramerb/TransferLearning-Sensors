import os,random,glob,sys
from scipy import stats as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DATA_ORI = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\originals\\PAMAP2\\Protocol\\*.dat'
SAVE_DIR = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\originalWindFreq\\'
init_freq = 100
n_classes = 4
act_translate = {}
act_translate[1] ='Pamap2-lying'
act_translate[2] ='Pamap2-sitting'
act_translate[3] ='Pamap2-standing'
act_translate[4] ='Pamap2-walking'
act_translate[5] ='Pamap2-running'
act_translate[7] ='Pamap2-nordic walking'
act_translate[12] ='Pamap2-ascending stairs'
act_translate[13] ='Pamap2-descending stairs'
act_translate[0] = 'dumb'
act_translate[6] = 'dumb'
act_translate[8] = 'dumb'
act_translate[9] = 'dumb'
act_translate[10] = 'dumb'
act_translate[11] = 'dumb'
act_translate[14] = 'dumb'
act_translate[15] = 'dumb'
act_translate[16] = 'dumb'
act_translate[17] = 'dumb'
act_translate[18] = 'dumb'
act_translate[19] = 'dumb'
act_translate[20] = 'dumb'
act_translate[21] = 'dumb'
act_translate[22] = 'dumb'
act_translate[23] = 'dumb'
act_translate[24] = 'dumb'



act_count = {}
act_count['Pamap2-walking'] = 0
act_count['Pamap2-lying'] = 0
act_count['Pamap2-ascending stairs'] = 0
act_count['Pamap2-descending stairs']= 0
act_count['dumb']= 0
act_count['Pamap2-sitting']= 0
act_count['Pamap2-standing']= 0
subjs = ['101','102','103','104','105','106','107','108']
final_folds = {}
for s in subjs:
	final_folds[s] = []

## IT IS WRONG SOMEHOW##
def process(desired_act,overlap = 0.5,new_freq =100,ts = 2 ):
	x  = []
	y = []
	sub_s = []
	allActivities = []
	sample_len = int(new_freq*ts)
	overlappingSize = sample_len - int(overlap * sample_len)

	for file in glob.glob(DATA_ORI):
		subject = file.split('\\')[-1].split('.')[0]
		print("\n" + subject)
		sub_s = []
		data = pd.read_table(file, header=None, sep='\s+')
		data = data.iloc[:, 1:13]
		data = data.drop([2,3,4, 5,6], axis=1)
		lines = data.to_numpy(dtype='float')
		curr_act = None
		for l in lines:
			act =int(l[0])
			if(curr_act != act and len(sub_s)>1) :
				if act in desired_act:
					my_act = act_translate[act]
					end = sample_len
					ini = 0
					while end <= len(sub_s):
						x.append(sub_s[ini:end])
						y.append(my_act)
						ini = ini + overlappingSize
						end = end +  overlappingSize
					# if (len(sub_s)%semple_len) !=0:
					# 	x.append(sub_s[0:semple_len])
					# 	y.append(my_act)
					sub_s = []
			if act in desired_act:
				sub_s.append(l[1:])
			curr_act = act
	dataX = np.array(x, dtype=float)
	dataY = np.array(y)
	return dataX,dataY


if __name__ == '__main__':
	windowSize = 2
	newFreq = 100
	overlapping = 0.5
	x = []
	y = []
	ini = 0
	# for i in [1,4,12,13]:
	desired_act = ['Walking', 'Ascending stairs', 'Descending stairs', 'Laying']
	x,y = process([1,4,12,13],overlap = overlapping, new_freq = newFreq,ts = windowSize)
		# x.append(aux)
		# y.append(np.array([act_translate[i]]*len(aux)))
		# ini_f = 0
		# for k,v in folds.items():
		# 	folds_idx = list(range(ini + ini_f,ini + v))
		# 	final_folds[k] = final_folds[k] + folds_idx
		# 	ini_f = v
		# ini = ini + len(aux)
		
	# folds = []
	# train =[]
	# for k,test in final_folds.items():
	# 	if len(test)>0:
	# 		for ki, vi in final_folds.items():
	# 			if ki != k and len(vi)>0:
	# 				train += vi
	# 		folds.append((train,test))
	# 		train =[]

	#outFile = os.path.join(SAVE_DIR,f'Pamap2_f{newFreq}_t{windowSize}_over{overlapping}_{n_classes}actv')
	outFile = os.path.join(SAVE_DIR, f'Pamap2AllOriginal_ovr')
	np.savez(outFile,X = x,y=y)
	print(x.shape)
	
