import numpy as np
import torch
import torch.nn as nn


class MMDLoss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMDLoss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		
		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2)
		if fix_sigma:
		    bandwidth = fix_sigma
		else:
		    bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)
	def rbf_kernel(self,source,target):
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


	def forward(self, source, target):
		batch_size = int(source.size()[0])
		kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss
	

class MMD_MSELoss(MMDLoss):
	def __init__(self, kernel_mul=2.0, kernel_num=5,alpha = .5):
		alpha = alpha
		super(MMD_MSELoss, self).__init__()
		return
	def forward(self,latentSource,latentTarget,sensor,rec):

		batch_size = int(latentSource.size()[0])
		# sourceIdx = np.where(domain==0)[0]
		# targetIdx = np.where(domain == 1)[0]
		# source =latent[sourceIdx]
		# target = latent[targetIdx]
		
		kernels = guassian_kernel(latentSource, latentTarget, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
		                          fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		lossMMD = torch.mean(XX + YY - XY - YX)
		mse= torch.nn.MSELoss(sensor,rec)
		return alpha*lossMMD + (1-self.alpha)*mse