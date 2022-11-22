import os,random,glob,sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DATA_ORI = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\originals\\PAMAP2\\Protocol\\*.dat'
DATA_DIR = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset4actv\\'
init_freq = 100
n_classes = 4
act_translate = {}
act_translate[1] ='Pamap2-lying'
act_translate[2] ='Pamap2-sitting'
act_translate[3] ='Pamap2-standing'
act_translate[4] ='Pamap2-walking'
act_translate[12] ='Pamap2-ascending stairs'
act_translate[13] ='Pamap2-descending stairs'

subjs = ['101','102','103','104','105','106','107','108','109']
final_folds = {}
for s in subjs:
	final_folds[s] = []


def process(my_act,overlap = 0.5,new_freq =100,ts = 2 ):
	x  = []
	y = []
	folds = {}
	step = int(init_freq/new_freq)
	sample_len = new_freq*ts
	
	ovs = int(overlap*sample_len)
	sub_s = []

	for file in glob.glob(DATA_ORI):
		subject = file.split('\\')[-1].split('.')[0][-3:]

		with open(file) as f:
			lines = f.readlines()
		i = 0
		for line in lines:
			if i%step ==0:
				#get the x axis of the second Acc
				l = line.split(' ')
				act =int(l[1])
				if act == my_act:
					s = np.array(l[7:13]).astype('float')
					if not np.isnan(s).any():
						sub_s.append(s)
					else:
						i = i-1
			i = i + 1
			if len(sub_s) == sample_len:
				x.append(sub_s)
				sub_s = sub_s[ovs:]
		sub_s = []
		folds[subject] = len(x)
	return np.array(x),folds


if __name__ == '__main__':

	x = []
	y = []
	ini = 0
	for i in [1,4,12,13]:
		aux,folds = process(my_act = i)
		x.append(aux)
		y.append(np.array([act_translate[i]]*len(aux)))
		ini_f = 0
		for k,v in folds.items():
			folds_idx = list(range(ini + ini_f,ini + v))
			final_folds[k] = final_folds[k] + folds_idx
			ini_f = v
		ini = ini + len(aux)
			
	data = np.concatenate(x,axis =0)[:,None,:,:]
	labels = np.concatenate(y,axis =0)
	folds = []
	train =[]
	for k,test in final_folds.items():
		if len(test)>0:
			for ki, vi in final_folds.items():
				if ki != k and len(vi)>0:
					train += vi
			folds.append((train,test))
			train =[]

	outFile = os.path.join(DATA_DIR,f'Pamap2_f100_t2_{n_classes}actv')
	np.savez(outFile,X = data,y=labels,folds = np.array(folds))
	
