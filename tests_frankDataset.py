from Dataset import Datasets
import numpy as np
import csv, sys, glob, os, pickle
import pandas as pd
from enum import Enum

from Dataset.Ucihar import UCIHAR,SignalsUCIHAR
from Dataset.Dsads import DSADS ,SignalsDsads
from Dataset.Uschad import USCHAD,SignalsUschad
from Dataset.Pamap2 import PAMAP2,SignalsPAMAP2

from Process.Manager import preprocess_datasets
from Process.Protocol import Loso

if __name__ == "__main__":
	
	if len(sys.argv) > 2:
		file_wisdm = sys.argv[1]
		dir_datasets = sys.argv[2]
		dir_save_file = sys.argv[3]
	else:
		source = 'C:\\Users\\gcram\\Documents\\Datasets\\originals\\'
		outPath = 'C:\\Users\\gcram\\Documents\\Datasets\\frankDataset\\'
	



	
	
	
	
