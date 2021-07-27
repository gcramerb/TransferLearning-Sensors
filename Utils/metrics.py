import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error as MSE
import scipy.stats as st
import sys
import pandas as pd
import json
import os


class absoluteMetrics():
	def __init__(self):
		self.a = 'a'
	
	def psnr_metric(self, original, compressed):
		mse = np.mean((original - compressed) ** 2)
		if mse == 0:  # MSE is zero means no noise is present in the signal .
			# Therefore PSNR have no importance.
			return 100
		max_pixel = 1
		psnr = 20 * log10(max_pixel / sqrt(mse))
		return psnr
	
	def psnr(self, xTrue, xRec):
		shape = xTrue.shape
		psnr = []
		psnrMean = []
		for i in range(shape[0]):
			for j in range(shape[-1]):
				psnr.append(self.psnr_metric(xTrue[i, :, j], xRec[i, :, j]))
			psnrMean.append(np.mean(psnr))
			psnr = []
		return np.mean(psnrMean)
	
	def myMSE(self, xTrue, xRec):
		# DH = dataHandler()
		# testRec, testTrue, idxAll = DH.get_reconstructed(dataset_name, miss, imp, si, path, file, fold_i)
		rmse_list = []
		# como aplicar vetorizacao nessa parte
		for i in range(len(xRec)):
			x = (xTrue[i, :, 0] - xRec[i, :, 0]) ** 2
			idx = np.where(x != 0)[0]
			rmse_list.append(MSE(xTrue[i, idx, 0], xRec[i, idx, 0], squared=False))
			rmse_list.append(MSE(xTrue[i, idx, 1], xRec[i, idx, 1], squared=False))
			rmse_list.append(MSE(xTrue[i, idx, 2], xRec[i, idx, 2], squared=False))
		return np.mean(rmse_list)
	
	def pearsonr(self, a, b):
		return 0
	
	def pearson_corr(self, xTrue, xRec):
		shape = xTrue.shape
		result = []
		for i in range(shape[0]):
			x = (xTrue[i, :, 0] - xRec[i, :, 0]) ** 2
			idx = np.where(x != 0)[0]
			x_corr = pearsonr(xTrue[i, idx, 0], xRec[i, idx, 0])
			y_corr = pearsonr(xTrue[i, idx, 1], xRec[i, idx, 1])
			z_corr = pearsonr(xTrue[i, idx, 2], xRec[i, idx, 2])
			result.append([x_corr, y_corr, z_corr])
		
		return np.mean(result, axis=0)
