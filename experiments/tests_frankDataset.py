from Dataset import Datasets
import numpy as np
import csv, sys, glob, os
import pandas as pd
from enum import Enum

from Dataset.Ucihar import UCIHAR,SignalsUCIHAR
from Dataset.Dsads import DSADS ,SignalsDsads
from Dataset.Uschad import USCHAD,SignalsUschad
from Dataset.Pamap2 import PAMAP2,SignalsPAMAP2

from Process.Manager import preprocess_datasets
from Process.Protocol import Loso

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--debug', action='store_true')
	args = parser.parse_args()
	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)
	if len(sys.argv) > 2:
		file_wisdm = sys.argv[1]
		dir_datasets = sys.argv[2]
		dir_save_file = sys.argv[3]
	else:
		source = 'C:\\Users\\gcram\\Documents\\Datasets\\originals\\'
		outPath = 'C:\\Users\\gcram\\Documents\\Datasets\\frankDataset\\'
	



	
	
	
	
