import numpy as np
import torch
import torch.nn as nn
import geomloss

from scipy.spatial import distance
from sklearn.preprocessing import normalize


class CenterLoss(nn.Module):
	"""Center loss.

	Reference:
	Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

	Args:
		num_classes (int): number of classes.
		feat_dim (int): feature dimension.
	"""
	
	def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu
		
		if self.use_gpu:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
	
	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
		          torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())
		
		classes = torch.arange(self.num_classes).long()
		if self.use_gpu: classes = classes.cuda()
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))
		
		dist = distmat * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
		
		return loss
	
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
		batch_size = int(latentSource.size()[0])

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

class CORAL(nn.Module):
	def __init__(self):
		super(CORAL, self).__init__()
		
	def forward(self,source, target):
		d = source.size(1)
		ns, nt = source.size(0), target.size(0)
		
		# source covariance
		tmp_s = torch.ones((1, ns)).cuda() @ source
		cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
		
		# target covariance
		tmp_t = torch.ones((1, nt)).cuda() @ target
		ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
		
		# frobenius norm
		loss = (cs - ct).pow(2).sum().sqrt()
		loss = loss / (4 * d * d)
		return loss

class OTLoss(nn.Module):
	def __init__(self,hyp = None, loss='sinkhorn',blur= 0.01,scaling= 0.7,debias = True):
		super(OTLoss, self).__init__()
		if hyp:
			blur = hyp['blur']
			scaling = hyp['scaling']
			debias = hyp['debias']
		self.lossFunc = geomloss.SamplesLoss(loss=loss,
		                                     p=2,
		                                     blur=blur,
		                                     reach=None,
		                                     diameter=None,
		                                     scaling=scaling,
		                                     truncate=5,
		                                     cost=None,
		                                     kernel=None,
		                                     cluster_scale=None,
		                                     debias=debias,
		                                     potentials=False,
		                                     verbose=False,
		                                     backend='auto')
	
	def forward(self, latentSource, latentTarget, label=None):
		return self.lossFunc(latentSource, latentTarget)

class myOTLoss(nn.Module):
	def __init__(self):
		self.dist = None
	
	@staticmethod
	def forward(ctx, mu, nu, dist, lam=1e-3, N=100):
		assert mu.dim() == 2 and nu.dim() == 2 and dist.dim() == 2
		bs = mu.size(0)
		d1, d2 = dist.size()
		assert nu.size(0) == bs and mu.size(1) == d1 and nu.size(1) == d2
		log_mu = mu.log()
		log_nu = nu.log()
		log_u = torch.full_like(mu, -math.log(d1))
		log_v = torch.full_like(nu, -math.log(d2))
		for i in range(N):
			log_v = sinkstep(dist, log_nu, log_u, lam)
			log_u = sinkstep(dist.t(), log_mu, log_v, lam)
		
		# this is slight abuse of the function. it computes (diag(exp(log_u))*Mt*exp(-Mt/lam)*diag(exp(log_v))).sum()
		# in an efficient (i.e. no bxnxm tensors) way in log space
		distances = (-sinkstep(-dist.log() + dist / lam, -log_v, log_u, 1.0)).logsumexp(1).exp()
		ctx.log_v = log_v
		ctx.log_u = log_u
		ctx.dist = dist
		ctx.lam = lam
		return distances
	
	@staticmethod
	def backward(ctx, grad_out):
		return grad_out[:, None] * ctx.log_u * ctx.lam, grad_out[:, None] * ctx.log_v * ctx.lam, None, None, None
	
class SinkhornDistance(torch.nn.Module):

	def __init__(self, eps, max_iter, reduction='sum'):
		super(SinkhornDistance, self).__init__()
		self.eps = eps
		self.max_iter = max_iter
		self.reduction = reduction

	def forward(self, mu, nu):
		s_mu = mu.shape[0]
		s_nu  = nu.shape[0]
		if s_mu!= s_nu:
			s = min(s_nu,s_mu)
			mu = mu[:s]
			nu = nu[:s]

		C = (torch.mean(mu,axis = 0)[None,:] - torch.mean(nu,axis = 0)[:,None])**2
		u = torch.zeros_like(mu)
		v = torch.zeros_like(nu)
		# To check if algorithm terminates because of threshold
		# or max iterations reached
		actual_nits = 0
		# Stopping criterion
		thresh = 1e-1

        # Sinkhorn iterations
		for i in range(self.max_iter):
			u1 = u  # useful to check the update
			u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
			v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
			err = (u - u1).abs().sum(-1).mean()
			
			actual_nits += 1
			if err.item() < thresh:
				break

		U, V = u, v
		# Transport plan pi = diag(a)*K*diag(b)
		pi = torch.exp(self.M(C, U, V))
		# Sinkhorn distance
		cost = torch.sum(pi * C, dim=(-2, -1))
		self.actual_nits = actual_nits
		if self.reduction == 'mean':
			cost = cost.mean()
		elif self.reduction == 'sum':
			cost = cost.sum()
		
		return cost

	def M(self, C, u, v):
		"Modified cost for logarithmic updates"
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
		return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
	
	@staticmethod
	def ave(u, u1, tau):
		"Barycenter subroutine, used by kinetic acceleration through extrapolation."
		return tau * u + (1 - tau) * u1

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
