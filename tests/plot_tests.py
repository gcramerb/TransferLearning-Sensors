
from Utils.visualization import plot_sensor
import os,random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

myActNames = {
	'walking': 0,
	'walking forward': 0,
	'ascending stairs': 1,
	'walking up': 1,
	'descending stairs': 2,
	'walking down': 2,
	'lying':3,
	'lying on back':3,
	'lying on right':3,
	'laying':3,
	'sleeping':3,
	'sitting': 4,
	'standing': 5,
}

datasetName = "Pamap2"
n_classes = 4
data_dir = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
file = os.path.join(data_dir, f'{datasetName}_f25_t2.npz')
with np.load(file, allow_pickle=True) as tmp:
	X = tmp['X'].astype('float32')
	y = tmp['y']
	folds = tmp['folds']
	
#act= f'{datasetName}-walking'
acts = ['Pamap2-walking','Pamap2-ascending stairs','Pamap2-descending stairs','Pamap2-lying']
idx = [i for i,v in enumerate(y) if v in acts]
X_fil = X[idx].copy()

idx = np.where(y==act)[0]
sample = 2
sensor = X[idx[sample]]
plot_sensor(sensor,act)




