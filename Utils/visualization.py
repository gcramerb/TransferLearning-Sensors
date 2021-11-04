import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import json
import seaborn as sn
import sys


#TODO : organizar essa função toda..

def plotReconstruction(rec,true, savePath = None, label = None,show=False,file = 'arqTest.png'):
	sensors = ['Accelerometer', ' Gyroscope', 'Magnetometer']
	n_sensors = rec.shape[0]
	sensors = sensors[0:n_sensors]
	axis = [' x', ' y', ' z']

	f, axarr = plt.subplots(2,n_sensors, sharex=True, sharey=False)

	# pyplot.figure()

	# plot total TRUE acc
	for i in range(n_sensors):
		axarr[0,i].plot(true[i,:, 0], color='green', label='eixo x')
		axarr[0,i].plot(true[i,:, 1], color='blue', label='eixo y')
		axarr[0,i].plot(true[i,:, 2], color='red', label='eixo z')
		axarr[0,i].set_title(f'{sensors[i]} Original')
		axarr[0,i].set_ylabel(f'value')
		axarr[0,i].legend()
		
		# plot total REconstructed sensor
		axarr[1,i].plot(rec[i,:, 0], color='green', label='eixo x')
		axarr[1,i].plot(rec[i,:, 1], color='blue', label='eixo y')
		axarr[1,i].plot(rec[i,:, 2], color='red', label='eixo z')
		axarr[1,i].set_title(f'{sensors[i]} Rconstructed')
		axarr[1,i].set_ylabel(f'value ')
		axarr[1,i].legend()
	
	if show:
		plt.show()
	else:
		plt.savefig(file)
		
	# plt.savefig(f"C:\\Users\gcram\Documents\Github\TCC\ + folder + '\' {label_file_name}.png")
	# file_name = path + f'/{label}_{tag}.png'
	
	# plt.savefig("../folder/%s_%s.png" % (label, file_name))
	plt.close()


	
def plot_cm(arqName = ''):
	infos = arqName.split('_')
	dataset = infos[1]
	missingRate = infos[2]
	algo = infos[3].split('.')[0]

	cm = np.load(arqName)['cm']
	cm = cm / cm.astype(np.float).sum(axis=1)
	labels = classesNames('USCHAD.npz')
	lab = [labels[x] for x in range(len(labels))]
	df_cm = pd.DataFrame(cm, index=lab, columns=lab)
	plt.figure(figsize=(10, 7))
	sn.heatmap(df_cm, annot=True,fmt='.2f',cmap='Blues')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title(f'Confusion matrix\n MDR: {missingRate} - recontruction: {algo}')
	plt.show()




def get_result():
	classifier = 'Sena'
	dataset = 'USCHAD'
	metric = 'F1'
	result = dict()
	for file in glob.glob(f"Catal/*.json"):
		with open(file) as json_file:
			data = json.load(json_file)
		infos  = file.split('_')
		missingRate = np.float(infos[2])
		method = infos[3].split('.')[0]
		if len(infos)> 4:
			method = method + '_' + infos[-1].split('.')[0]
		
		try:
			result[method][missingRate] = data[metric]
		except:
			result[method] = dict()
			result[method][missingRate] = data[metric]
	
	df = pd.DataFrame(result)
	df = df.sort_index()
	df = df * 100
	df.index = df.index * 100
	return df

