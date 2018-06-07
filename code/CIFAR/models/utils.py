import numpy as np
import math

import torch
import torch.nn as nn
from models.layers import set_layer_from_config


def cuda_available():
	return torch.cuda.is_available()


def list_sum(x):
	if len(x) == 1:
		return x[0]
	elif len(x) == 2:
		return x[0] + x[1]
	else:
		return x[0] + list_sum(x[1:])


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class Cutout(object):
	"""Randomly mask out one or more patches from an image.
	
	please refer to https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
	Args:
	    n_holes (int): Number of patches to cut out of each image.
	    length (int): The length (in pixels) of each square patch.
	"""
	
	def __init__(self, n_holes, length):
		self.n_holes = n_holes
		self.length = length
	
	def __call__(self, img):
		"""
		Args:
			img (Tensor): Tensor image of size (C, H, W).
		Returns:
			Tensor: Image with n_holes of dimension length x length cut out of it.
		"""
		if isinstance(img, np.ndarray):
			h = img.shape[1]
			w = img.shape[2]
		else:
			h = img.size(1)
			w = img.size(2)
		
		mask = np.ones((h, w), np.float32)
		
		for n in range(self.n_holes):
			# center point of the cutout region
			y = np.random.randint(h)
			x = np.random.randint(w)
			
			width = int(self.length / 2)
			y1 = np.clip(y - width, 0, h)
			y2 = np.clip(y + width, 0, h)
			x1 = np.clip(x - width, 0, w)
			x2 = np.clip(x + width, 0, w)
			
			mask[y1: y2, x1: x2] = 0.0
		
		if isinstance(img, np.ndarray):
			mask = np.expand_dims(mask, axis=0)
		else:
			mask = torch.from_numpy(mask)
			mask = mask.expand_as(img)
		
		return img * mask


class TransitionBlock(nn.Module):
	def __init__(self, layers):
		super(TransitionBlock, self).__init__()
		
		self.layers = nn.ModuleList(layers)
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	
	def get_config(self):
		return {
			'name': TransitionBlock.__name__,
			'layers': [
				layer.get_config() for layer in self.layers
			]
		}
	
	@staticmethod
	def set_from_config(config):
		layers = []
		for layer_config in config.get('layers'):
			layer = set_layer_from_config(layer_config)
			layers.append(layer)
		block = TransitionBlock(layers)
		return block
	
	def virtual_forward(self, x, init=False):
		for layer in self.layers:
			x = layer.virtual_forward(x, init)
		return x
	
	def claim_ready(self, nBatch, noise=None):
		for layer in self.layers:
			layer.claim_ready(nBatch, noise)


class BasicBlockWiseConvNet(nn.Module):
	def __init__(self, blocks, classifier):
		super(BasicBlockWiseConvNet, self).__init__()
		
		self.blocks = nn.ModuleList(blocks)
		self.classifier = classifier
	
	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		x = x.view(x.size(0), -1)  # flatten
		x = self.classifier(x)
		return x
	
	@property
	def building_block(self):
		raise NotImplementedError
	
	def get_config(self):
		raise NotImplementedError
	
	@staticmethod
	def set_from_config(config):
		raise NotImplementedError
	
	@staticmethod
	def set_standard_net(**kwargs):
		raise NotImplementedError
	
	def init_model(self, model_init, init_div_groups):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if model_init == 'he_fout':
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					if init_div_groups:
						n /= m.groups
					m.weight.data.normal_(0, math.sqrt(2. / n))
				elif model_init == 'he_fin':
					n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
					if init_div_groups:
						n /= m.groups
					m.weight.data.normal_(0, math.sqrt(2. / n))
				else:
					raise NotImplementedError
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					m.bias.data.zero_()
	
	def set_non_ready_layers(self, data_loader, nBatch, noise=None, print_info=True):
		self.eval()
		
		top1 = AverageMeter()
		data_loader.drop_last = True
		batch_num = 0
		batch_size = 0
		while batch_num < nBatch:
			for _input, target in data_loader:
				if torch.cuda.is_available():
					target = target.cuda(async=True)
					_input = _input.cuda()
				input_var = torch.autograd.Variable(_input, volatile=True)
				
				x = input_var
				for block in self.blocks:
					x = block.virtual_forward(x, init=(batch_num == 0))
				x = x.view(x.size(0), -1)  # flatten
				x = self.classifier(x)
				
				acc1 = accuracy(x.data, target, topk=(1,))
				top1.update(acc1[0][0], _input.size(0))
				
				if batch_size == 0:
					batch_size = x.size()[0]
				else:
					assert batch_size == x.size()[0]
				batch_num += 1
				if batch_num >= nBatch:
					break
		if print_info: print(top1.avg)
		for block in self.blocks:
			block.claim_ready(nBatch, noise)
	
	def mimic_run_with_linear_regression(self, data_loader, src_model, sample_size=None,
	                                     distill_epochs=0, distill_lr=0.1, print_info=True):
		src_model.eval()
		input_images, model_outputs = [], []
		for _input, _ in data_loader:
			input_images.append(_input)
			if next(src_model.parameters()).is_cuda:
				_input = _input.cuda()
			input_var = torch.autograd.Variable(_input, volatile=True)
			
			final_logit = src_model(input_var)
			model_outputs.append(final_logit.data.cpu())
			
			if sample_size:
				sample_size -= _input.size(0)
				if sample_size <= 0:
					break
		
		criterion = nn.MSELoss()
		if torch.cuda.is_available():
			criterion = criterion.cuda()
		self.train()
		
		if distill_epochs > 0:
			optimizer = torch.optim.SGD(self.parameters(), lr=distill_lr)
			losses = AverageMeter()
			if print_info: print('start distilling')
			for _j in range(distill_epochs):
				lr = distill_lr
				rand_indexes = np.random.permutation(len(input_images))
				for _i, idx in enumerate(rand_indexes):
					# T_total = distill_epochs * len(rand_indexes)
					# T_cur = _j * len(rand_indexes) + _i
					# lr = 0.5 * distill_lr * (1 + math.cos(math.pi * T_cur / T_total))
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr
					
					_input = input_images[idx]
					target = model_outputs[idx]
					if torch.cuda.is_available():
						input_var = torch.autograd.Variable(_input.cuda())
						target_var = torch.autograd.Variable(target.cuda())
					else:
						input_var = torch.autograd.Variable(_input)
						target_var = torch.autograd.Variable(target)
					self_model_out = self.forward(input_var)
					loss = criterion(self_model_out, target_var)
					
					losses.update(loss.data[0], _input.size(0))
					
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				if print_info: print('epoch [%d]: %f\tlr=%f' % (_j, losses.avg, lr))

