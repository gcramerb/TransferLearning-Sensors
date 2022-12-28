import numpy as np
import pandas as pd
import glob, os, time
from enum import Enum
from scipy.io import loadmat


actNameUschad = {
    1:  'Walking Forward',
    2:  'Walking Left',
    3:  'Walking Right',
    4:  'Walking Up',
    5:  'Walking Down',
    6:  'Running',
    7:  'Jumping',
    8:  'Sitting',
    9:  'Standing',
    10: 'Sleeping',
    11: 'Elevator Up',
    12: 'Elevator Down',
}
fixUSCHADNames = {
	'walking-left':'walking',
	'walking-right':'walking',
	'walking-forward':'walking',
	'walking-up':'ascending stairs',
	'walking-down':'descending stairs',
	'sleeping':'lying',
	'walking-downstairs':'descending stairs',
	'walking-upstairs':'ascending stairs',
	'walk-forward':'walking',
	'walk-left':'walking',
	'walk-right':'walking',
	'walk-up':'ascending stairs',
	'walk-down':'descending stairs',
	'walk-upstairs':'ascending stairs',
	'walk-downstairs':'descending stairs',
}

class SignalsUschad(Enum):
    acc_front_right_hip_X = 0
    acc_front_right_hip_Y = 1
    acc_front_right_hip_Z = 2
    gyr_front_right_hip_X = 3
    gyr_front_right_hip_Y = 4
    gyr_front_right_hip_Z = 5


init_freq = 100


class USCHAD():
	def __init__(self,  overlap, new_freq, ts):
		
		self.new_freq = new_freq
		self.windowSize = ts
		self.sample_len = new_freq* ts
		self.initialFreq = 100
		self.overlappingSize = self.sample_len - int(overlap*self.sample_len)
		self.dir_dataset = "C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\originals\\USC-HAD\\"
		self.dir_save_file = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\originalWindFreq\\'
		sig_usc = [SignalsUschad.acc_front_right_hip_X, SignalsUschad.acc_front_right_hip_Y,
		           SignalsUschad.acc_front_right_hip_Z]
		sig_usc += [SignalsUschad.gyr_front_right_hip_X, SignalsUschad.gyr_front_right_hip_Y,
		            SignalsUschad.gyr_front_right_hip_Z]
		self.signals_use = sig_usc
		self.desired_act = ['walking-left','walking-right','walking-forward', 'walking-up', 'walking-down', 'sleeping','walking-downstairs','walking-upstairs','walk-forward', 'walk-left', 'walk-right', 'walk-up','walk-down','walk-upstairs', 'walk-downstairs']
		self.dataX = []
		self.dataY = []
		self.allActvities = []
	

	def preprocess(self):
		mat_files = []
		for root, dirs, files in os.walk(self.dir_dataset):
			if len(dirs) == 0:
				mat_files.extend([os.path.join(root, f) for f in files])
		
		for filepath in mat_files:
			mat_file = loadmat(filepath)
			act = mat_file['activity'][0]
			subject = int(mat_file['subject'][0])
			trial_id = int(mat_file['trial'][0])
			trial_data = mat_file['sensor_readings'].astype('float64')
			self.allActvities.append(act)
			if act.lower() in self.desired_act:
				end = self.sample_len
				ini = 0
				while end <= len(trial_data):
					self.dataX.append(trial_data[ini:end,:])
					self.dataY.append("Uschad-" + fixUSCHADNames[act])
					ini = ini + self.overlappingSize
					end = end + + self.overlappingSize

		self.dataX = np.array(self.dataX, dtype=float)
		self.dataY = np.array(self.dataY)
		np.savez_compressed(os.path.join(self.dir_save_file, "UschadAllOriginal_ovr"),
		                    X=self.dataX,
		                    y=self.dataY)


if __name__ == '__main__':
	windowSize = 5
	newFreq = 100
	overlapping = 0.5
	x = []
	y = []
	dat = USCHAD( overlapping, newFreq, windowSize)
	dat.preprocess()
# data = np.concatenate(x, axis=0)[:, None, :, :]
# labels = np.concatenate(y, axis=0)
# outFile = os.path.join(DATA_DIR, f'Ucihar_f{newFreq}_t{windowSize}_over{overlapping}_{n_classes}actv')
# np.savez(outFile, X=data, y=labels, folds=np.array(folds))

