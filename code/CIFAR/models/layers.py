import numpy as np

import torch.nn as nn
import torch


def get_layer_by_name(name):
	if name == ConvLayer.__name__:
		return ConvLayer
	elif name == DepthConvLayer.__name__:
		return DepthConvLayer
	elif name == PoolingLayer.__name__:
		return PoolingLayer
	elif name == IdentityLayer.__name__:
		return IdentityLayer
	elif name == LinearLayer.__name__:
		return LinearLayer
	else:
		raise ValueError('unrecognized layer: %s' % name)


def set_layer_from_config(layer_config):
	layer_name = layer_config.pop('name')
	layer = get_layer_by_name(layer_name)
	layer = layer(**layer_config)
	return layer


def apply_noise(weights, noise=None):
	if noise is None:
		return weights
	else:
		assert isinstance(noise, dict)
	
	noise_type = noise.get('type', 'normal')
	if noise_type == 'normal':
		ratio = noise.get('ratio', 1e-3)
		std = torch.std(weights)
		weights_noise = torch.Tensor(weights.size()).normal_(0, std * ratio)
	elif noise_type == 'uniform':
		ratio = noise.get('ratio', 1e-3)
		_min, _max = torch.min(weights), torch.max(weights)
		width = (_max - _min) / 2 * ratio
		weights_noise = torch.Tensor(weights.size()).uniform_(-width, width)
	else:
		raise NotImplementedError
	if weights.is_cuda:
		weights_noise = weights_noise.cuda()
	return weights + weights_noise


class BasicLayer(nn.Module):
	def __init__(self, in_channels, out_channels,
	             use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(BasicLayer, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		self.use_bn = use_bn
		self.act_func = act_func
		self.dropout_rate = dropout_rate
		self.ops_order = ops_order
		
		self.layer_ready = layer_ready
		
		""" batch norm, activation, dropout """
		if self.use_bn:
			if self.bn_before_weight:
				self.bn = nn.BatchNorm2d(in_channels)
			else:
				self.bn = nn.BatchNorm2d(out_channels)
		else:
			self.bn = None
		if act_func == 'relu':
			if self.ops_list[0] == 'act':
				self.activation = nn.ReLU(inplace=False)
			else:
				self.activation = nn.ReLU(inplace=True)
		else:
			self.activation = None
		if self.dropout_rate > 0:
			self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
		else:
			self.dropout = None
	
	@property
	def ops_list(self):
		return self.ops_order.split('_')
	
	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)
	
	@property
	def bn_before_act(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'act':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)
	
	def forward(self, x):
		for op in self.ops_list:
			if op == 'weight':
				x = self.weight_call(x)
			elif op == 'bn':
				if self.bn is not None:
					x = self.bn(x)
			elif op == 'act':
				if self.activation is not None:
					x = self.activation(x)
			else:
				raise ValueError('Unrecognized op: %s' % op)
		if self.dropout is not None:
			x = self.dropout(x)
		return x
	
	def weight_call(self, x):
		raise NotImplementedError
	
	def get_same_padding(self, kernel_size):
		if kernel_size == 1:
			padding = 0
		elif kernel_size == 3:
			padding = 1
		elif kernel_size == 5:
			padding = 2
		elif kernel_size == 7:
			padding = 3
		else:
			raise NotImplementedError
		return padding
	
	def get_config(self):
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'use_bn': self.use_bn,
			'act_func': self.act_func,
			'dropout_rate': self.dropout_rate,
			'ops_order': self.ops_order,
		}
	
	def copy_bn(self, copy_layer, noise=None):
		if noise is None: noise = {}
		if self.use_bn:
			copy_layer.bn.weight.data = self.bn.weight.data.clone()
			copy_layer.bn.bias.data = self.bn.bias.data.clone()
			copy_layer.bn.running_mean = self.bn.running_mean.clone()
			copy_layer.bn.running_var = self.bn.running_var.clone()
	
	def copy(self, noise=None):
		raise NotImplementedError
	
	def split(self, split_list, noise=None):
		raise NotImplementedError
	
	@property
	def get_str(self):
		raise NotImplementedError
	
	def virtual_forward(self, x, init=False):
		if not self.layer_ready:
			if self.use_bn:
				if init:
					self.bn.running_mean.zero_()
					self.bn.running_var.zero_()
				if self.bn_before_act:
					x_ = x
				else:
					x_ = self.activation(x)
				batch_mean = x_
				for dim in [0, 2, 3]:
					batch_mean = torch.mean(batch_mean, dim=dim, keepdim=True)
				batch_var = (x_ - batch_mean) * (x_ - batch_mean)
				for dim in [0, 2, 3]:
					batch_var = torch.mean(batch_var, dim=dim, keepdim=True)
				batch_mean = torch.squeeze(batch_mean)
				batch_var = torch.squeeze(batch_var)
				
				self.bn.running_mean += batch_mean.data
				self.bn.running_var += batch_var.data
			return x
		else:
			return self.forward(x)
	
	def claim_ready(self, nBatch):
		if not self.layer_ready:
			if self.use_bn:
				self.bn.running_mean /= nBatch
				self.bn.running_var /= nBatch
				self.bn.bias.data = self.bn.running_mean.clone()
				self.bn.weight.data = torch.sqrt(self.bn.running_var + self.bn.eps)
			self.layer_ready = True


class ConvLayer(BasicLayer):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
	             use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(ConvLayer, self).__init__(in_channels, out_channels,
		                                use_bn, act_func, dropout_rate, ops_order, layer_ready)
		
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		
		padding = self.get_same_padding(self.kernel_size)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
		                      padding=padding, dilation=self.dilation, groups=self.groups, bias=self.bias)
	
	def weight_call(self, x):
		x = self.conv(x)
		return x
	
	def get_config(self):
		config = {
			'name': ConvLayer.__name__,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dilation': self.dilation,
			'groups': self.groups,
			'bias': self.bias,
		}
		config.update(super(ConvLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		conv_copy = set_layer_from_config(self.get_config())
		# copy weights
		conv_copy.conv.weight.data = apply_noise(self.conv.weight.data.clone(), noise.get('wider'))
		if self.bias:
			conv_copy.conv.bias.data = apply_noise(self.conv.bias.data.clone(), noise.get('wider'))
		self.copy_bn(conv_copy, noise.get('bn'))
		return conv_copy
	
	def split(self, split_list, noise=None):
		assert np.sum(split_list) == self.out_channels
		if noise is None: noise = {}
		
		seg_layers = []
		if self.groups == 1:
			for seg_size in split_list:
				seg_config = self.get_config()
				seg_config['out_channels'] = seg_size
				seg_layers.append(set_layer_from_config(seg_config))
	
			_pt = 0
			for _i in range(len(split_list)):
				seg_size = split_list[_i]
				seg_layers[_i].conv.weight.data = self.conv.weight.data.clone()[_pt:_pt + seg_size, :, :, :]
				if self.bias:
					seg_layers[_i].conv.bias.data = self.conv.bias.data.clone()[_pt:_pt + seg_size]
				if self.use_bn:
					if self.bn_before_weight:
						self.copy_bn(seg_layers[_i], noise.get('bn'))
					else:
						seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
						seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
						seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
						seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
				_pt += seg_size
		else:
			assert self.groups % len(split_list) == 0
			assert np.all([split_list[0] == split_list[_i] for _i in range(1, len(split_list))])
			
			new_groups = self.groups // len(split_list)
			for seg_size in split_list:
				seg_config = self.get_config()
				seg_config['out_channels'] = seg_size
				seg_config['in_channels'] = self.in_channels // len(split_list)
				seg_config['groups'] = new_groups
				seg_layers.append(set_layer_from_config(seg_config))
			
			in_pt, out_pt = 0, 0
			for _i in range(len(split_list)):
				in_seg_size = self.in_channels // len(split_list)
				out_seg_size = split_list[_i]
				seg_layers[_i].conv.weight.data = self.conv.weight.data.clone()[out_pt:out_pt + out_seg_size, :, :, :]
				if self.bias:
					seg_layers[_i].conv.bias.data = self.conv.bias.data.clone()[out_pt:out_pt + out_seg_size]
				if self.use_bn:
					if self.bn_before_weight:
						seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[in_pt:in_pt + in_seg_size]
						seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[in_pt:in_pt + in_seg_size]
						seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[in_pt:in_pt + in_seg_size]
						seg_layers[_i].bn.running_var = self.bn.running_var.clone()[in_pt:in_pt + in_seg_size]
					else:
						seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[out_pt:out_pt + out_seg_size]
						seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[out_pt:out_pt + out_seg_size]
						seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[out_pt:out_pt + out_seg_size]
						seg_layers[_i].bn.running_var = self.bn.running_var.clone()[out_pt:out_pt + out_seg_size]
				out_pt += out_seg_size
				in_pt += in_seg_size
		return seg_layers
	
	@property
	def get_str(self):
		if self.groups == 1:
			return '%dx%d_Conv' % (self.kernel_size, self.kernel_size)
		else:
			return '%dx%d_GroupConv' % (self.kernel_size, self.kernel_size)
	
	def virtual_forward(self, x, init=False):
		if not self.layer_ready and self.bias:
			assert self.ops_order == 'bn_act_weight'
			if init:
				self.conv.bias.data.zero_()
			min_val = x
			for dim in [0, 2, 3]:
				min_val, _ = torch.min(min_val, dim=dim, keepdim=True)
			min_val = torch.squeeze(min_val)
			self.conv.bias.data = torch.min(self.conv.bias.data, min_val.data)
		return super(ConvLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		if noise is None: noise = {}
		if not self.layer_ready:
			super(ConvLayer, self).claim_ready(nBatch)
			if self.bias:
				self.bn.bias.data -= self.conv.bias.data
			
			mid = self.kernel_size // 2
			self.conv.weight.data.zero_()
			weight_init = torch.cat([
				torch.eye(self.conv.weight.size(1)) for _ in range(self.conv.groups)
			], dim=0)
			self.conv.weight.data[:, :, mid, mid] = apply_noise(weight_init, noise.get('deeper'))
		
		assert self.layer_ready


class DepthConvLayer(BasicLayer):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
	             use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(DepthConvLayer, self).__init__(in_channels, out_channels,
		                                     use_bn, act_func, dropout_rate, ops_order, layer_ready)
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		
		padding = self.get_same_padding(self.kernel_size)
		self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, stride=self.stride,
		                            padding=padding, dilation=self.dilation, groups=in_channels, bias=False)
		self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=self.bias)
	
	def weight_call(self, x):
		x = self.depth_conv(x)
		x = self.point_conv(x)
		return x
	
	def get_config(self):
		config = {
			'name': DepthConvLayer.__name__,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dilation': self.dilation,
			'groups': self.groups,
			'bias': self.bias,
		}
		config.update(super(DepthConvLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		depth_conv_copy = set_layer_from_config(self.get_config())
		# copy weights
		depth_conv_copy.depth_conv.weight.data = apply_noise(self.depth_conv.weight.data.clone(), noise.get('wider'))
		depth_conv_copy.point_conv.weight.data = apply_noise(self.point_conv.weight.data.clone(), noise.get('wider'))
		if self.bias:
			depth_conv_copy.point_conv.bias.data = apply_noise(self.point_conv.bias.data.clone(), noise.get('wider'))
		self.copy_bn(depth_conv_copy, noise.get('bn'))
		return depth_conv_copy
	
	def split(self, split_list, noise=None):
		if noise is None: noise = {}
		assert np.sum(split_list) == self.out_channels
		
		seg_layers = []
		for seg_size in split_list:
			seg_config = self.get_config()
			seg_config['out_channels'] = seg_size
			seg_layers.append(set_layer_from_config(seg_config))
		
		_pt = 0
		for _i in range(len(split_list)):
			seg_size = split_list[_i]
			seg_layers[_i].depth_conv.weight.data = apply_noise(self.depth_conv.weight.data.clone(), noise.get('wider'))
			seg_layers[_i].point_conv.weight.data = self.point_conv.weight.data.clone()[_pt:_pt + seg_size, :, :, :]
			if self.bias:
				seg_layers[_i].point_conv.bias.data = self.point_conv.bias.data.clone()[_pt:_pt + seg_size]
			if self.use_bn:
				if self.bn_before_weight:
					self.copy_bn(seg_layers[_i], noise.get('bn'))
				else:
					seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
					seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
					seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
					seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
			_pt += seg_size
		return seg_layers
	
	@property
	def get_str(self):
		return '%dx%d_DepthConv' % (self.kernel_size, self.kernel_size)
	
	def virtual_forward(self, x, init=False):
		if not self.layer_ready and self.bias:
			assert self.ops_order == 'bn_act_weight'
			if init:
				self.point_conv.bias.data.zero_()
			min_val = x
			for dim in [0, 2, 3]:
				min_val, _ = torch.min(min_val, dim=dim, keepdim=True)
			min_val = torch.squeeze(min_val)
			self.point_conv.bias.data = torch.min(self.point_conv.bias.data, min_val.data)
		return super(DepthConvLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		if noise is None: noise = {}
		if not self.layer_ready:
			super(DepthConvLayer, self).claim_ready(nBatch)
			if self.bias:
				self.bn.bias.data -= self.point_conv.bias.data
			
			mid = self.kernel_size // 2
			self.depth_conv.weight.data.zero_()
			self.depth_conv.weight.data[:, 0, mid, mid].fill_(1)
			self.depth_conv.weight.data = apply_noise(self.depth_conv.weight.data, noise.get('deeper'))
			
			self.point_conv.weight.data.zero_()
			self.point_conv.weight.data[:, :, 0, 0] = torch.eye(self.point_conv.weight.size(0))
			self.point_conv.weight.data = apply_noise(self.point_conv.weight.data, noise.get('deeper'))
		
		assert self.layer_ready


class PoolingLayer(BasicLayer):
	def __init__(self, in_channels, out_channels, pool_type, kernel_size=2, stride=2,
	             use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(PoolingLayer, self).__init__(in_channels, out_channels,
		                                   use_bn, act_func, dropout_rate, ops_order, layer_ready)
		
		self.pool_type = pool_type
		self.kernel_size = kernel_size
		self.stride = stride
		
		if self.stride == 1:
			padding = self.get_same_padding(self.kernel_size)
		else:
			padding = 0
		
		if self.pool_type == 'avg':
			self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False)
		elif self.pool_type == 'max':
			self.pool = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
		else:
			raise NotImplementedError
	
	def weight_call(self, x):
		x = self.pool(x)
		return x
	
	def get_config(self):
		config = {
			'name': PoolingLayer.__name__,
			'pool_type': self.pool_type,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
		}
		config.update(super(PoolingLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		copy_layer = set_layer_from_config(self.get_config())
		self.copy_bn(copy_layer, noise.get('bn'))
		return copy_layer
	
	def split(self, split_list, noise=None):
		if noise is None: noise = {}
		assert np.sum(split_list) == self.out_channels
		
		seg_layers = []
		for seg_size in split_list:
			seg_config = self.get_config()
			seg_config['in_channels'] = seg_size
			seg_config['out_channels'] = seg_size
			seg_layers.append(set_layer_from_config(seg_config))
		
		_pt = 0
		for _i in range(len(split_list)):
			seg_size = split_list[_i]
			if self.use_bn:
				seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
			_pt += seg_size
		return seg_layers
	
	@property
	def get_str(self):
		return '%dx%d_%sPool' % (self.kernel_size, self.kernel_size, self.pool_type.upper())
	
	def virtual_forward(self, x, init=False):
		return super(PoolingLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		super(PoolingLayer, self).claim_ready(nBatch)
		assert self.layer_ready


class IdentityLayer(BasicLayer):
	def __init__(self, in_channels, out_channels,
	             use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(IdentityLayer, self).__init__(in_channels, out_channels,
		                                    use_bn, act_func, dropout_rate, ops_order, layer_ready)
	
	def weight_call(self, x):
		return x
	
	def get_config(self):
		config = {
			'name': IdentityLayer.__name__,
		}
		config.update(super(IdentityLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		copy_layer = set_layer_from_config(self.get_config())
		self.copy_bn(copy_layer, noise.get('bn'))
		return copy_layer
	
	def split(self, split_list, noise=None):
		if noise is None: noise = {}
		assert np.sum(split_list) == self.out_channels
		
		seg_layers = []
		for seg_size in split_list:
			seg_config = self.get_config()
			seg_config['in_channels'] = seg_size
			seg_config['out_channels'] = seg_size
			seg_layers.append(set_layer_from_config(seg_config))
		
		_pt = 0
		for _i in range(len(split_list)):
			seg_size = split_list[_i]
			if self.use_bn:
				seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
			_pt += seg_size
		return seg_layers
	
	@property
	def get_str(self):
		return 'Identity'
	
	def virtual_forward(self, x, init=False):
		return super(IdentityLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		super(IdentityLayer, self).claim_ready(nBatch)
		assert self.layer_ready


class LinearLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(LinearLayer, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias
		
		self.linear = nn.Linear(self.in_features, self.out_features, self.bias)
	
	def forward(self, x):
		return self.linear(x)
	
	def get_config(self):
		return {
			'name': LinearLayer.__name__,
			'in_features': self.in_features,
			'out_features': self.out_features,
			'bias': self.bias,
		}
