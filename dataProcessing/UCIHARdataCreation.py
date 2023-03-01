import numpy as np
import pandas as pd
import glob, os, time
from enum import Enum


class SignalsUcihar(Enum):
	acc_body_X = 0
	acc_body_Y = 1
	acc_body_Z = 2
	gyr_body_X = 3
	gyr_body_Y = 4
	gyr_body_Z = 5
	acc_total_X = 6
	acc_total_Y = 7
	acc_total_Z = 8


actNameUcihar = {
	1: 'Walking',
	2: 'Ascending stairs',
	3: 'Descending stairs',
	4: 'Sitting',
	5: 'Standing',
	6: 'Laying'
}
actNameUciharGeneric = {
	1: 'Ucihar-walking',
	2: 'Ucihar-ascending stairs',
	3: 'Ucihar-descending stairs',
	4: 'Ucihar-sitting',
	5: 'Ucihar-standing',
	6: 'Ucihar-lying'
}
class SignalsUcihar(Enum):
	acc_body_X = 0
	acc_body_Y = 1
	acc_body_Z = 2
	gyr_body_X = 3
	gyr_body_Y = 4
	gyr_body_Z = 5
	acc_total_X = 6
	acc_total_Y = 7
	acc_total_Z = 8

class UCIHAR():
	def __init__(self, overlap, new_freq, ts ):
		self.overlap = overlap
		self.new_freq = new_freq
		self.windowSize = ts
		self.initialFreq = 50
		self.dir_dataset = "C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\originals\\uci-human-activity-recognition\\original\\"
		self.dir_save_file ='C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset6actv\\'
		sig_uci = [SignalsUcihar.acc_body_X, SignalsUcihar.acc_body_Y, SignalsUcihar.acc_body_Z]
		sig_uci += [SignalsUcihar.gyr_body_X, SignalsUcihar.gyr_body_Y, SignalsUcihar.gyr_body_Z]
		self.signals_use = sig_uci
		self.desired_act = ['Walking','Ascending stairs','Descending stairs','Sitting','Standing','Laying']
		self.dataX = []
		self.dataY = []

	def print_info(self):
		return "device:  smartphone (Samsung Galaxy S II)" \
		       "frequency: 50 Hz" \
		       "positions: body" \
		       "sensors: accelerometer, gyroscope" \
		       "subjects: 30" \
		       "Age: 19-48" \
		       "example: https://www.youtube.com/watch?v=XOEN9W05_4A" \
		       "Obs: Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration."
	
	def preprocess(self):
		start = time.time()
		trial_id = np.ones([30]).astype(int)
		for part in ['train', 'test']:
			data = []
			path = os.path.join(self.dir_dataset,"UCI HAR Dataset",part, 'Inertial Signals')
			for sig in ['body_acc', 'body_gyro']:
				for axis in ['x', 'y', 'z']:
					data.append(
						pd.read_csv(os.path.join(path, f'{sig}_{axis}_{part}.txt'), delim_whitespace=True, header=None))
			
			#subjects = pd.read_csv(os.path.join(self.dir_dataset, 'original', f'subject_{part}.csv'))
			labels = pd.read_csv(os.path.join(self.dir_dataset, f'y_{part}.csv'))
			for i in range(len(labels)):
				act = actNameUcihar[labels.iloc[i].values[0]]
				if act in self.desired_act:
					trial = None
					for d in data:
						if trial is not None:
							trial = np.concatenate([trial, np.expand_dims(d.iloc[i, :].values, 1)], axis=1)
						else:
							trial = np.expand_dims(d.iloc[i, :].values, 1)
					
					signals = [signal.value for signal in self.signals_use]
					trial = trial[:, signals]
					self.dataX.append(trial)
					self.dataY.append(actNameUciharGeneric[labels.iloc[i].values[0]])

		self.dataX = np.array(self.dataX, dtype=float)
		self.dataY = np.array(self.dataY)
		np.savez_compressed(os.path.join(self.dir_save_file, "Ucihar_6activities"),
		                    X=self.dataX,
		                    y=self.dataY)


if __name__ == '__main__':
	windowSize = 2.56
	newFreq = 50
	overlapping = 0.0
	x = []
	y = []
	ini = 0
	dat = UCIHAR(overlapping, newFreq, windowSize)
	dat.preprocess()
	# data = np.concatenate(x, axis=0)[:, None, :, :]
	# labels = np.concatenate(y, axis=0)
	# outFile = os.path.join(DATA_DIR, f'Ucihar_f{newFreq}_t{windowSize}_over{overlapping}_{n_classes}actv')
	# np.savez(outFile, X=data, y=labels, folds=np.array(folds))

