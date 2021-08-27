import numpy as np
import pandas as pd
myActNames = {
	'walking': 0,
	'walking forward': 0,
	'ascending stairs': 1,
	'walking up': 1,
	'descending stairs': 2,
	'walking down': 2,
	'sitting':3,
	'standing':4,
	'lying':5,
	'lying on back':5,
	'lying on right':5,
	'laying':5,
	'sleeping':5
}

def categorical_to_int(y):
	y1 = list(map(lambda x: x.split('-')[-1], y))
	y2 = np.array(list(map(lambda x: myActNames[x], y1)))
	y3 = np.zeros([len(y),len(pd.unique(y))])
	for i in range(len(y3)):
		y3[i,y2[i]] = 1
	return y3
