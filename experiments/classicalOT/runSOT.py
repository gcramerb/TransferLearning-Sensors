DATA_DIR = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\frankDataset\\'
datasetList = ['Dsads','Ucihar','Uschad','Pamap2']
n_classes = 4
import sys

# from geomloss import SamplesLoss
sys.path.insert(0, '../')

from dataProcessing.dataModule import SingleDatasetModule
from train.trainerTL_pl import TLmodel

from literature.SOT import SOT


def run():
	source = 'Ucihar'
	target = 'Uschad'
	trainParams = {}
	trainParams['input_shape'] = (2,50,3)
	
	PATH_MODEL = f'../../saved/TLmodel{source}_to_{target}_ot.ckpt'
	
	dm_source = SingleDatasetModule(data_dir=DATA_DIR, datasetName=source, n_classes=n_classes,
	                                input_shape=trainParams['input_shape'], batch_size=128)
	dm_source.setup(Loso = False)
	dm_target = SingleDatasetModule(data_dir=DATA_DIR, datasetName=target, n_classes=n_classes,
	                                input_shape=trainParams['input_shape'], batch_size=128)
	dm_target.setup(Loso = False,split = True)
	new_model = TLmodel.load_from_checkpoint(PATH_MODEL)
	predictions = new_model.predict(dm_source, dm_target)
	
	trueSource = predictions['trueSource']
	predSource = predictions['predSource']
	latentSource = predictions['latentSource']
	trueTarget = predictions['trueTarget']
	predTarget = predictions['predTarget']
	latentTarget = predictions['latentTarget']
	return latentSource,trueSource,latentTarget,trueTarget

if __name__ == '__main__':

	tsot = SOT('ACT', './clustertemp/', 19, 0.5, 1, 3)
	Sx, Sy, Tx, Ty = run()
	spath = './data/test_MDA_JCPOT_ACT_diag_SG.json'
	tpath = './clustertemp/data/test_MDA_JCPOT_ACT_19_diag_TG.json'
	tmodelpath = './clustertemp/model/test_MDA_JCPOT_ACT_19_diag_H'
	
	pred, acc = tsot.fit_predict(Sx, Sy, Tx, Ty, spath, 'D', tpath, tmodelpath, 'H')
	print(acc)