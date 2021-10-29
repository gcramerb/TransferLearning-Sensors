import numpy as np
import torch
import torch.nn as nn
import geomloss

from scipy.spatial import distance
from sklearn.preprocessing import normalize


class MMDLoss(nn.Module):
	def __init__(self, kernel_mul=2.0, kernel_num=5):
		super(MMDLoss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0]) + int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		
		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0 - total1) ** 2).sum(2)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)
	
	def rbf_kernel(self, source, target):
		xx, yy, zz = torch.mm(source, source.t()), torch.mm(target, target.t()), torch.mm(source, target.t())
		rx = (xx.diag().unsqueeze(0).expand_as(xx))
		ry = (yy.diag().unsqueeze(0).expand_as(yy))
		
		dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
		dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
		dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
		
		XX, YY, XY = (torch.zeros(xx.shape).to(device),
		              torch.zeros(xx.shape).to(device),
		              torch.zeros(xx.shape).to(device))
		
		bandwidth_range = [10, 15, 20, 50]
		for a in bandwidth_range:
			XX += torch.exp(-0.5 * dxx / a)
			YY += torch.exp(-0.5 * dyy / a)
			XY += torch.exp(-0.5 * dxy / a)
		return XX, YY, XY
	
	def forward(self, latentSource, latentTarget, kernel='gaussian'):
		penalty = 0
		batch_size = int(latent.size()[0])

		if kernel == 'gaussian':
			xx = torch.mean(self.guassian_kernel(latentSource,latentSource, kernel_mul=self.kernel_mul,
			                                     kernel_num=self.kernel_num, fix_sigma=self.fix_sigma))
			yy = torch.mean(self.guassian_kernel(latentTarget, latentTarget, kernel_mul=self.kernel_mul,
			                                     kernel_num=self.kernel_num, fix_sigma=self.fix_sigma))
			xy = torch.mean(self.guassian_kernel(latentSource, latentTarget, kernel_mul=self.kernel_mul,
			                                     kernel_num=self.kernel_num, fix_sigma=self.fix_sigma))
			# penalty = torch.mean(XX + YY - XY -YX)
			mmd = xx + yy - 2 * xy
			
			return mmd
		elif kernel == 'rbf':
			xx, yy, xy = self.rbf_kernel(latentSource,latentTarget)
			return torch.mean(XX + YY - 2. * XY)


class OTLoss(nn.Module):
	def __init__(self, loss='sinkhorn', p=2, blur=0.05, scaling=0.9):
		super(OTLoss, self).__init__()
		self.loss = loss
		self.p = p
		self.blur = blur
		self.scaling = scaling
		self.lossFunc = geomloss.SamplesLoss(loss=self.loss, p=self.p, blur=self.blur, reach=None, diameter=None,
		                                     scaling=self.scaling,
		                                     truncate=5, cost=None, kernel=None, cluster_scale=None, debias=True,
		                                     potentials=False, verbose=False, backend='auto')
	
	def forward(self, latentSource, latentTarget, label=None):
		return self.lossFunc(latentSource, latentTarget)


class classDistance(nn.Module):
	def __init__(self):
		super(classDistance, self).__init__()
		self.eps = 1e-10
		self.cos_sim = nn.CosineSimilarity(dim = 1,eps=self.eps)

	
	def forward(self, latent, label):
		"""
		First we are going to aply distance between classes
		:param latent:
		:param domain:
		:param label:
		:return:
		"""

		classes = label.cpu().numpy()
		avail_labels = np.unique(classes)
		# Compute indixes of embeddings for each class.
		class_positions = []
		for lab in avail_labels:
			class_positions.append(np.where(classes == lab)[0])
		
		# Compute average intra-class distance and center of mass.
		com_class, dists_class = [], []
		for i in range(len(class_positions)):
			for j in range(i+1,len(class_positions)):
				x1 = latent[class_positions[i],:,None]
				x2 =  latent[class_positions[j]].t()[None,:,:]
				sim_mt = self.cos_sim(x1,x2)
				# dists = torch.abs(1 - sim_mt).clamp(min=eps)
				l_ =  len(sim_mt) ** 2
				l__ = len(sim_mt)
				d = l_ - l__
				d = max(1,d)
				dists = torch.sum(torch.sum(sim_mt)) /d
				dists_class.append(dists)
				# com = torch.mean(features[class_pos], axis=0).reshape(1, -1)
				# com = nn.functional.normalize(com).reshape(-1)
				# com_class.append(com)
		
		# Compute mean inter-class distances by the class-coms.
		# com_class = torch.stack(com_class)
		# mean_inter_sim =  cos_sim(com_class, com_class)
		# mean_inter_dist = torch.abs(1 - mean_inter_sim).clamp(min=eps)
		# mean_inter_dist = torch.sum(mean_inter_dist) / (len(mean_inter_dist) ** 2 - len(mean_inter_dist))
		#
		# Compute distance ratio

		dists_class = torch.stack(dists_class)

		res = torch.mean(dists_class)
		return res
	
#		return torch.mean(dists_class / mean_inter_dist)
