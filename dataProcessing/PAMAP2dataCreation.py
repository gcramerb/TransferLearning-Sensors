import os,random,glob,sys
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
act_translate[7] ='Pamap2-walking'
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
subjs = ['101','102','103','104','105','106','107','108','109']
final_folds = {}
for s in subjs:
	final_folds[s] = []


def process(desired_act,overlap = 0.5,new_freq =100,ts = 2 ):
	x  = []
	y = []
	folds = {}
	step = int(init_freq/new_freq)
	sample_len = new_freq*ts
	
	ovs = sample_len - int(overlap*sample_len)
	sub_s = []

	for file in glob.glob(DATA_ORI):
		subject = file.split('\\')[-1].split('.')[0][-3:]
		sub_s = []
		with open(file) as f:
			lines = f.readlines()
		i = 0
		curr_act = None
		for line in lines:
			
			#if i%step ==0:
			#get the x axis of the second Acc
			l = line.split(' ')
			act =int(l[1])
			if(curr_act != act):
				sub_s = []
				curr_act = act
			#if act in desired_act:
			s = np.array(l[7:13]).astype('float')
			sub_s.append(s)
			my_act = act_translate[act]
			#act_count[my_act] +=1
				# if not np.isnan(s).any():
				# 	sub_s.append(s)
				# else:
				# 	i = i-1
			#i = i + 1
			if len(sub_s) == sample_len and act in desired_act:
				x.append(sub_s)
				y.append(my_act)
				sub_s = sub_s[ovs:]

		
		# folds[subject] = len(x)
	dataX = np.array(x, dtype=float)
	dataY = np.array(y)
	return dataX,dataY


if __name__ == '__main__':
	windowSize = 5
	newFreq = 100
	overlapping = 0
	x = []
	y = []
	ini = 0
	# for i in [1,4,12,13]:
	desired_act = ['Walking', 'Ascending stairs', 'Descending stairs', 'Laying']
	x,y = process([1,4,7,12,13],overlap = overlapping, new_freq = newFreq,ts = windowSize)
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
	outFile = os.path.join(SAVE_DIR, f'Pamap2AllOriginal')
	np.savez(outFile,X = data,y=labels)
	
