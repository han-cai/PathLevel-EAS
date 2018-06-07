from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LearnedGroupConv(nn.Module):
	global_progress = 0.0
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
	             padding=0, dilation=1, groups=1,
	             condense_factor=None, dropout_rate=0.,
	             only_conv=False, bias=False):
		super(LearnedGroupConv, self).__init__()
		self.only_conv = only_conv
		if not self.only_conv:
			self.norm = nn.BatchNorm2d(in_channels)
			self.relu = nn.ReLU(inplace=True)
		self.dropout_rate = dropout_rate
		if self.dropout_rate > 0:
			self.drop = nn.Dropout(dropout_rate, inplace=False)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
		                      padding, dilation, groups=1, bias=False)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.groups = groups
		self.condense_factor = condense_factor
		if self.condense_factor is None:
			self.condense_factor = self.groups
		# Parameters that should be carefully used
		self.register_buffer('_count', torch.zeros(1))
		self.register_buffer('_stage', torch.zeros(1))
		self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
		# Check if arguments are valid
		assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
		assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
		assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"
	
	def forward(self, x):
		self._check_drop()
		if not self.only_conv:
			x = self.norm(x)
			x = self.relu(x)
		if self.dropout_rate > 0:
			x = self.drop(x)
		# Masked output
		weight = self.conv.weight * self.mask
		return F.conv2d(x, weight, None, self.conv.stride,
		                self.conv.padding, self.conv.dilation, 1)
	
	def _check_drop(self):
		progress = LearnedGroupConv.global_progress
		delta = 0
		# Get current stage
		for i in range(self.condense_factor - 1):
			if progress * 2 < (i + 1) / (self.condense_factor - 1):
				stage = i
				break
		else:
			stage = self.condense_factor - 1
		# Check for dropping
		if not self._at_stage(stage):
			self.stage = stage
			delta = self.in_channels // self.condense_factor
		if delta > 0:
			self._dropping(delta)
		return
	
	def _dropping(self, delta):
		weight = self.conv.weight * self.mask
		# Sum up all kernels
		# Assume only apply to 1x1 conv to speed up
		assert weight.size()[-1] == 1
		weight = weight.abs().squeeze()
		assert weight.size()[0] == self.out_channels
		assert weight.size()[1] == self.in_channels
		d_out = self.out_channels // self.groups
		# Shuffle weight
		weight = weight.view(d_out, self.groups, self.in_channels)
		weight = weight.transpose(0, 1).contiguous()
		weight = weight.view(self.out_channels, self.in_channels)
		# Sort and drop
		for i in range(self.groups):
			wi = weight[i * d_out:(i + 1) * d_out, :]
			# Take corresponding delta index
			di = wi.sum(0).sort()[1][self.count:self.count + delta]
			for d in di.data:
				self._mask[i::self.groups, d, :, :].fill_(0)
		self.count = self.count + delta
	
	@property
	def count(self):
		return int(self._count[0])
	
	@count.setter
	def count(self, val):
		self._count.fill_(val)
	
	@property
	def stage(self):
		return int(self._stage[0])
	
	@stage.setter
	def stage(self, val):
		self._stage.fill_(val)
	
	@property
	def mask(self):
		return Variable(self._mask)
	
	def _at_stage(self, stage):
		return (self._stage == stage).all()
	
	@property
	def lasso_loss(self):
		if self._at_stage(self.groups - 1):
			# no lasso loss in the optimization stage
			return 0
		weight = self.conv.weight * self.mask
		# Assume only apply to 1x1 conv to speed up
		assert weight.size()[-1] == 1
		weight = weight.squeeze().pow(2)
		d_out = self.out_channels // self.groups
		# Shuffle weight
		weight = weight.view(d_out, self.groups, self.in_channels)
		weight = weight.sum(0).clamp(min=1e-6).sqrt()
		return weight.sum()


class LearnedLinear(nn.Module):
	global_progress = 0.0
	
	def __init__(self, in_features, out_features, drop_rate=0.5):
		super(LearnedLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.drop_rate = drop_rate  # 0 means no drop and 1 means full drop
		
		self.linear = nn.Linear(self.in_features, self.out_features)
		
		# Parameters that should be carefully used
		self.register_buffer('_stage', torch.zeros(1))
		self.register_buffer('_mask', torch.ones(self.linear.weight.size()))
	
	def forward(self, x):
		self._check_drop()
		weight = self.linear.weight * self.mask
		return F.linear(x, weight, self.linear.bias)
	
	def _check_drop(self):
		progress = LearnedLinear.global_progress
		if progress < 1 / 2:
			stage = 0
		else:
			stage = 1
		# Check for dropping
		if not self._at_stage(stage):
			self.stage = stage
			_, index = self.linear.weight.data.abs().sum(0).sort()
			if self.drop_rate > 0:
				index = index[:int(self.linear.in_features * self.drop_rate)]  # to be masked
				for _i in index:
					self._mask[:, _i].fill_(0)
					self.linear.weight.data[:, _i].fill_(0)
	
	def get_masked_linear(self):
		for _i in range(self.in_features):
			if (self._mask[:, _i] == 0).all():
				self.linear.weight.data[:, _i].fill_(0)
		return self.linear
	
	@property
	def stage(self):
		return int(self._stage[0])
	
	@stage.setter
	def stage(self, val):
		self._stage.fill_(val)
	
	@property
	def mask(self):
		return Variable(self._mask)
	
	def _at_stage(self, stage):
		return (self._stage == stage).all()
	

def ShuffleLayer(x, groups):
	batchsize, num_channels, height, width = x.data.size()
	channels_per_group = num_channels // groups
	# reshape
	x = x.view(batchsize, groups,
	           channels_per_group, height, width)
	# transpose
	x = torch.transpose(x, 1, 2).contiguous()
	# flatten
	x = x.view(batchsize, -1, height, width)
	return x


class CondensingLinear(nn.Module):
	def __init__(self, model, drop_rate=0.5):
		super(CondensingLinear, self).__init__()
		self.in_features = int(model.in_features * drop_rate)
		self.out_features = model.out_features
		self.linear = nn.Linear(self.in_features, self.out_features)
		self.register_buffer('index', torch.LongTensor(self.in_features))
		_, index = model.weight.data.abs().sum(0).sort()
		index = index[model.in_features - self.in_features:]
		self.linear.bias.data = model.bias.data.clone()
		for i in range(self.in_features):
			self.index[i] = index[i]
			self.linear.weight.data[:, i] = model.weight.data[:, index[i]]
	
	def forward(self, x):
		x = torch.index_select(x, 1, Variable(self.index))
		x = self.linear(x)
		return x


class CondenseLinear(nn.Module):
	def __init__(self, in_features, out_features, drop_rate=0.5):
		super(CondenseLinear, self).__init__()
		self.in_features = int(in_features * drop_rate)
		self.out_features = out_features
		self.linear = nn.Linear(self.in_features, self.out_features)
		self.register_buffer('index', torch.LongTensor(self.in_features))
	
	def forward(self, x):
		x = torch.index_select(x, 1, Variable(self.index))
		x = self.linear(x)
		return x


class CondensingConv(nn.Module):
	def __init__(self, model):
		super(CondensingConv, self).__init__()
		self.in_channels = model.conv.in_channels \
		                   * model.groups // model.condense_factor
		self.out_channels = model.conv.out_channels
		self.groups = model.groups
		self.condense_factor = model.condense_factor
		self.only_conv = model.only_conv
		if not self.only_conv:
			self.norm = nn.BatchNorm2d(self.in_channels)
			self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(self.in_channels, self.out_channels,
		                      kernel_size=model.conv.kernel_size,
		                      padding=model.conv.padding,
		                      groups=self.groups,
		                      bias=False,
		                      stride=model.conv.stride)
		self.register_buffer('index', torch.LongTensor(self.in_channels))
		index = 0
		mask = model._mask.mean(-1).mean(-1)
		for i in range(self.groups):
			for j in range(model.conv.in_channels):
				if index < (self.in_channels // self.groups) * (i + 1) \
						and mask[i, j] == 1:
					for k in range(self.out_channels // self.groups):
						idx_i = int(k + i * (self.out_channels // self.groups))
						idx_j = index % (self.in_channels // self.groups)
						self.conv.weight.data[idx_i, idx_j, :, :] = \
							model.conv.weight.data[int(i + k * self.groups), j, :, :]
						if not self.only_conv:
							self.norm.weight.data[index] = model.norm.weight.data[j]
							self.norm.bias.data[index] = model.norm.bias.data[j]
							self.norm.running_mean[index] = model.norm.running_mean[j]
							self.norm.running_var[index] = model.norm.running_var[j]
					self.index[index] = j
					index += 1
	
	def forward(self, x):
		x = torch.index_select(x, 1, Variable(self.index))
		if not self.only_conv:
			x = self.norm(x)
			x = self.relu(x)
		x = self.conv(x)
		x = ShuffleLayer(x, self.groups)
		return x


class CondenseConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
	             stride=1, padding=0, groups=1,
	             only_conv=False):
		super(CondenseConv, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.groups = groups
		
		self.only_conv = only_conv
		if not self.only_conv:
			self.norm = nn.BatchNorm2d(self.in_channels)
			self.relu = nn.ReLU(inplace=True)
			
		self.conv = nn.Conv2d(self.in_channels, self.out_channels,
		                      kernel_size=kernel_size,
		                      stride=stride,
		                      padding=padding,
		                      groups=self.groups,
		                      bias=False)
		self.register_buffer('index', torch.LongTensor(self.in_channels))
		self.index.fill_(0)
	
	def forward(self, x):
		x = torch.index_select(x, 1, Variable(self.index))
		if not self.only_conv:
			x = self.norm(x)
			x = self.relu(x)
		x = self.conv(x)
		x = ShuffleLayer(x, self.groups)
		return x


class Conv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
	             stride=1, padding=0, groups=1,
	             bias=False, has_shuffle=False):
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
		
		self.groups = groups
		self.has_shuffle = has_shuffle
		super(Conv, self).__init__()
		self.norm = nn.BatchNorm2d(in_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(in_channels, out_channels,
		                      kernel_size=kernel_size,
		                      stride=stride,
		                      padding=padding, bias=bias,
		                      groups=groups)
	
	def forward(self, x):
		x = self.norm(x)
		x = self.relu(x)
		x = self.conv(x)
		if self.groups > 1 and self.has_shuffle:
			x = ShuffleLayer(x, self.groups)
		return x


class DepthseparableConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
	             stride=1, bias=False, groups=1, has_shuffle=False):
		super(DepthseparableConv, self).__init__()
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
		
		self.groups = groups
		self.has_shuffle = has_shuffle
		
		self.norm = nn.BatchNorm2d(in_channels)
		self.relu = nn.ReLU(inplace=True)
		self.depth_conv = nn.Conv2d(in_channels, in_channels,
		                            kernel_size=kernel_size,
		                            stride=stride,
		                            padding=padding, bias=False,
		                            groups=in_channels)
		self.point_conv = nn.Conv2d(in_channels, out_channels,
		                            kernel_size=1,
		                            bias=bias, groups=groups)
	
	def forward(self, x):
		x = self.norm(x)
		x = self.relu(x)
		x = self.depth_conv(x)
		x = self.point_conv(x)
		if self.groups > 1 and self.has_shuffle:
			x = ShuffleLayer(x, self.groups)
		return x


class Identity(nn.Module):
	def forward(self, x):
		return x
	
	
def torch_list_sum(x):
	if len(x) == 1:
		return x[0]
	elif len(x) == 2:
		return x[0] + x[1]
	else:
		return x[0] + torch_list_sum(x[1:])


class TreeNode(nn.Module):
	def __init__(self, edge_configs, child_nodes, in_channels, out_channels,
	             split_type='copy', merge_type='add', use_avg=False,
	             path_drop=0, drop_only_add=False, use_zero_drop=True,
	             bn_before_add=False):
		super(TreeNode, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.split_type = split_type
		self.merge_type = merge_type
		self.use_avg = use_avg
		self.path_drop = path_drop
		self.drop_only_add = drop_only_add
		self.use_zero_drop = use_zero_drop
		self.bn_before_add = bn_before_add
		
		assert len(edge_configs) == len(child_nodes)
		self.child_num = len(edge_configs)
		
		if self.split_type == 'copy':
			in_dim_list = [self.in_channels] * self.child_num
		elif self.split_type == 'split':
			in_dim_list = self.get_split_list(self.in_channels, self.child_num)
		else:
			assert self.child_num == 1
			in_dim_list = [self.in_channels]
		self.in_dim_list = in_dim_list
		
		if self.merge_type == 'add':
			out_dim_list = [self.out_channels] * self.child_num
		elif self.merge_type == 'concat':
			out_dim_list = self.get_split_list(self.out_channels, self.child_num)
		else:
			assert self.child_num == 1
			out_dim_list = [self.out_channels]
		self.out_dim_list = out_dim_list
		
		edges = []
		branch_bns = []
		for idx, edge_config in enumerate(edge_configs):
			child_in, child_out = in_dim_list[idx], out_dim_list[idx]
			edge2child = self.build_edge(edge_config, child_in, child_out)
			edges.append(edge2child)
			if self.bn_before_add and self.merge_type == 'add':
				branch_bns.append(nn.BatchNorm2d(child_out))
		self.edges = nn.ModuleList(edges)
		self.child_nodes = nn.ModuleList(child_nodes)
		if len(branch_bns) > 0:
			self.branch_bns = nn.ModuleList(branch_bns)
		else:
			self.branch_bns = []
	
	def forward(self, x):
		if self.split_type == 'copy':
			child_inputs = [x] * self.child_num
		elif self.split_type == 'split':
			child_inputs, _pt = [], 0
			in_dim_list = self.get_split_list(x.size(1), self.child_num)
			for seg_size in in_dim_list:
				seg_x = x[:, _pt:_pt+seg_size, :, :].contiguous()
				child_inputs.append(seg_x)
				_pt += seg_size
		else:
			child_inputs = [x]
			
		child_outputs = []
		for _i in range(self.child_num):
			# child_out = child_inputs[_i].contiguous()
			child_out = child_inputs[_i]
			child_out = self.path_drop_forward(child_out, _i)
			child_outputs.append(child_out)
		
		if self.merge_type == 'concat':
			output = torch.cat(child_outputs, dim=1)
		elif self.merge_type == 'add':
			output = torch_list_sum(child_outputs)
			if self.use_avg:
				output /= self.child_num
		else:
			assert len(child_outputs) == 1
			output = child_outputs[0]
		return output
	
	@staticmethod
	def path_normal_forward(x, edge, child, branch_bn, use_avg=False):
		if edge:
			x = edge(x)
		edge_x = x
		if child:
			x = child(x)
		if branch_bn:
			x = branch_bn(x)
			x += edge_x
			if use_avg: x /= 2
		return x
	
	def path_drop_forward(self, x, branch_idx):
		edge = self.edges[branch_idx]
		child = self.child_nodes[branch_idx]
		branch_bn = self.branch_bns[branch_idx] if len(self.branch_bns) > 0 else None
		
		if self.drop_only_add and self.merge_type != 'add':
			apply_drop = False
		else:
			apply_drop = True
		if not self.use_zero_drop and (edge.__dict__.get('out_channels', 1) != edge.__dict__.get('in_channels', 1)):
			apply_drop = False
		if apply_drop and self.path_drop > 0:
			p = random.uniform(0, 1)
			drop_flag = p < self.path_drop
			if self.training:
				# train
				if self.use_zero_drop:
					if drop_flag:
						batch_size = x.size()[0]
						feature_map_size = x.size()[2:4]
						stride = edge.__dict__.get('stride', 1)
						out_channels = self.out_dim_list[branch_idx]
						padding = torch.zeros(batch_size, out_channels,
						                      feature_map_size[0] // stride, feature_map_size[1] // stride)
						if x.is_cuda:
							padding = padding.cuda()
						path_out = torch.autograd.Variable(padding)
					else:
						path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
						path_out /= (1 - self.path_drop)
				else:
					raise NotImplementedError
			else:
				if self.use_zero_drop:
					path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
				else:
					raise NotImplementedError
		else:
			path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
		return path_out
	
	@staticmethod
	def get_split_list(in_dim, child_num):
		in_dim_list = [in_dim // child_num] * child_num
		for _i in range(in_dim % child_num):
			in_dim_list[_i] += 1
		return in_dim_list
	
	def build_edge(self, edge_config, in_channels, out_channels):
		edge_type = edge_config.pop('type')
		if edge_type == LearnedGroupConv.__name__:
			return LearnedGroupConv(in_channels=in_channels, out_channels=out_channels, **edge_config)
		elif edge_type == Conv.__name__:
			return Conv(in_channels=in_channels, out_channels=out_channels, has_shuffle=True, **edge_config)
		elif edge_type == DepthseparableConv.__name__:
			return DepthseparableConv(in_channels=in_channels, out_channels=out_channels, has_shuffle=True,
			                          **edge_config)
		elif edge_type == Identity.__name__:
			return Identity()
		elif edge_type == 'Pool':
			op, kernel_size = edge_config['op'], edge_config['kernel_size']
			if kernel_size == 3:
				padding = 1
			elif kernel_size == 5:
				padding = 2
			elif kernel_size == 7:
				padding = 3
			else:
				raise NotImplementedError
			if op == 'avg':
				return nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)
			elif op == 'max':
				return nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError
			

