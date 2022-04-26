import numpy as np
import pandas as pd

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
# for softLabel situation. It es
myActNames2 = {
	0:'PS-walking',
	1:'PS-ascending stairs',
	2:'PS-descending stairs',
	3:'PS-lying',
	4:'PS-sitting',
	5:'PS-standing',
}

def categorical_to_int(y):
	y1 = list(map(lambda x: x.split('-')[-1], y))
	y2 = np.array(list(map(lambda x: myActNames[x], y1)))
	y3 = np.zeros([len(y),len(pd.unique(y))])
	for i in range(len(y3)):
		y3[i,y2[i]] = np.long(1)
	return y3

def int2categorical(y):
	return list(map(lambda x: myActNames2[x], y))

