import random

import torch
import torch.nn as nn

from models.layers import set_layer_from_config, IdentityLayer, PoolingLayer, ConvLayer
from models.utils import list_sum


class TreeNode(nn.Module):
	
	SET_MERGE_TYPE = 'set_merge_type'
	INSERT_NODE = 'insert_node'
	REPLACE_IDENTITY_EDGE = 'replace_identity_edge'
	
	def __init__(self, child_nodes, edges, in_channels, out_channels,
	             split_type='copy', merge_type='add', use_avg=True, bn_before_add=False,
	             path_drop_rate=0, use_zero_drop=True, drop_only_add=False, cell_drop_rate=0):
		super(TreeNode, self).__init__()
		
		self.edges = nn.ModuleList(edges)
		self.child_nodes = nn.ModuleList(child_nodes)
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.split_type = split_type
		self.merge_type = merge_type
		
		self.use_avg = use_avg
		self.bn_before_add = bn_before_add
		
		self.path_drop_rate = path_drop_rate
		self.use_zero_drop = use_zero_drop
		self.drop_only_add = drop_only_add
		self.cell_drop_rate = cell_drop_rate
		
		assert len(edges) == len(child_nodes)
		
		self.branch_bns = None
		if self.bn_before_add and self.merge_type == 'add':
			branch_bns = []
			for _i in range(self.child_num):
				branch_bns.append(nn.BatchNorm2d(self.out_dim_list[_i]))
			self.branch_bns = nn.ModuleList(branch_bns)
	
	@property
	def child_num(self):
		return len(self.edges)
	
	@property
	def in_dim_list(self):
		if self.split_type == 'copy':
			in_dim_list = [self.in_channels] * self.child_num
		elif self.split_type == 'split':
			in_dim_list = self.get_split_list(self.in_channels, self.child_num)
		else:
			assert self.child_num == 1
			in_dim_list = [self.in_channels]
		return in_dim_list
	
	@property
	def out_dim_list(self):
		if self.merge_type == 'add':
			out_dim_list = [self.out_channels] * self.child_num
		elif self.merge_type == 'concat':
			out_dim_list = self.get_split_list(self.out_channels, self.child_num)
		else:
			assert self.child_num == 1
			out_dim_list = [self.out_channels]
		return out_dim_list
		
	@staticmethod
	def get_split_list(in_dim, child_num):
		in_dim_list = [in_dim // child_num] * child_num
		for _i in range(in_dim % child_num):
			in_dim_list[_i] += 1
		return in_dim_list
	
	@staticmethod
	def path_normal_forward(x, edge=None, child=None, branch_bn=None, use_avg=False):
		if edge is not None:
			x = edge(x)
		edge_x = x
		if child is not None:
			x = child(x)
		if branch_bn is not None:
			x = branch_bn(x)
			x += edge_x
			if use_avg: x /= 2
		return x
	
	def path_drop_forward(self, x, branch_idx):
		edge, child = self.edges[branch_idx], self.child_nodes[branch_idx]
		branch_bn = None if self.branch_bns is None else self.branch_bns[branch_idx]
		if self.drop_only_add and self.merge_type != 'add':
			apply_drop = False
		else:
			apply_drop = True
		if (edge.in_channels != edge.out_channels or edge.__dict__.get('stride', 1) > 1) and not self.use_zero_drop:
			apply_drop = False
		if apply_drop and self.path_drop_rate > 0:
			p = random.uniform(0, 1)
			drop_flag = p < self.path_drop_rate
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
						path_out /= (1 - self.path_drop_rate)
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
	
	def forward(self, x, virtual=False, init=False):
		if self.cell_drop_rate > 0:
			if self.training:
				p = random.uniform(0, 1)
				drop_flag = p < self.cell_drop_rate
				if self.use_zero_drop:
					if drop_flag:
						# drop
						batch_size = x.size()[0]
						feature_map_size = x.size()[2:4]
						stride = self.edges[0].__dict__.get('stride', 1)
						padding = torch.zeros(batch_size, self.out_channels,
						                      feature_map_size[0] // stride, feature_map_size[1] // stride)
						if x.is_cuda:
							padding = padding.cuda()
						return torch.autograd.Variable(padding)
					else:
						# not drop
						backup = self.cell_drop_rate
						self.cell_drop_rate = 0
						output = self.forward(x, virtual, init)  # normal forward
						self.cell_drop_rate = backup
						output /= 1 - self.cell_drop_rate
						return output
				else:
					raise NotImplementedError
			else:
				if self.use_zero_drop:
					pass  # normal forward
				else:
					raise NotImplementedError
				
		if self.split_type == 'copy':
			child_inputs = [x] * self.child_num
		elif self.split_type == 'split':
			child_inputs, _pt = [], 0
			for seg_size in self.in_dim_list:
				seg_x = x[:, _pt:_pt+seg_size, :, :].contiguous()
				child_inputs.append(seg_x)
				_pt += seg_size
		else:
			child_inputs = [x]
		
		child_outputs = []
		for branch_idx in range(self.child_num):
			if virtual:
				edge, child = self.edges[branch_idx], self.child_nodes[branch_idx]
				branch_bn = None if self.branch_bns is None else self.branch_bns[branch_idx]
				path_out = child_inputs[branch_idx]
				if edge:
					path_out = edge.virtual_forward(path_out, init)
				if child:
					path_out = child.virtual_forward(path_out, init)
				if branch_bn:
					if init:
						branch_bn.running_mean.zero_()
						branch_bn.running_var.zero_()
					x_ = path_out
					batch_mean = x_
					for dim in [0, 2, 3]:
						batch_mean = torch.mean(batch_mean, dim=dim, keepdim=True)
					batch_var = (x_ - batch_mean) * (x_ - batch_mean)
					for dim in [0, 2, 3]:
						batch_var = torch.mean(batch_var, dim=dim, keepdim=True)
					batch_mean = torch.squeeze(batch_mean)
					batch_var = torch.squeeze(batch_var)
					
					branch_bn.running_mean += batch_mean.data
					branch_bn.running_var += batch_var.data
					# path_out = branch_bn(path_out)
			else:
				path_out = self.path_drop_forward(child_inputs[branch_idx], branch_idx)
			child_outputs.append(path_out)

		if self.merge_type == 'concat':
			output = torch.cat(child_outputs, dim=1)
		elif self.merge_type == 'add':
			output = list_sum(child_outputs)
			if self.use_avg:
				output /= self.child_num
		else:
			assert len(child_outputs) == 1
			output = child_outputs[0]
		return output
	
	def get_config(self):
		child_configs = []
		for child in self.child_nodes:
			if child is None:
				child_configs.append(None)
			else:
				child_configs.append(child.get_config())
		edge_configs = []
		for edge in self.edges:
			if edge is None:
				edge_configs.append(None)
			else:
				edge_configs.append(edge.get_config())
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'split_type': self.split_type,
			'merge_type': self.merge_type,
			'use_avg': self.use_avg,
			'bn_before_add': self.bn_before_add,
			'path_drop_rate': self.path_drop_rate,
			'use_zero_drop': self.use_zero_drop,
			'drop_only_add': self.drop_only_add,
			'cell_drop_rate': self.cell_drop_rate,
			'edges': edge_configs,
			'child_nodes': child_configs,
		}
	
	@staticmethod
	def set_from_config(config):
		child_nodes = []
		for child_config in config.pop('child_nodes'):
			if child_config is None:
				child_nodes.append(None)
			else:
				child_nodes.append(TreeNode.set_from_config(child_config))
		edges = []
		for edge_config in config.pop('edges'):
			if edge_config is None:
				edges.append(None)
			else:
				edges.append(set_layer_from_config(edge_config))
		return TreeNode(child_nodes=child_nodes, edges=edges, **config)

	def get_node(self, node_path):
		node = self
		for branch in node_path:
			node = node.child_nodes[branch]
		return node
	
	def apply_transformation(self, node_path, op_type, op_param):
		tree_node = self.get_node(node_path)
		
		if op_type == TreeNode.SET_MERGE_TYPE:
			tree_node.set_merge_type(**op_param)
		elif op_type == TreeNode.INSERT_NODE:
			tree_node.insert_node(**op_param)
		elif op_type == TreeNode.REPLACE_IDENTITY_EDGE:
			tree_node.replace_identity_edge(**op_param)
		else:
			raise NotImplementedError
	
	@property
	def get_str(self):
		if self.child_num > 0:
			children_str = []
			for _i, child in enumerate(self.child_nodes):
				child_str = None if child is None else child.get_str
				children_str.append('%s=>%s' % (self.edges[_i].get_str, child_str))
			children_str = '[%s]' % ', '.join(children_str)
		else:
			children_str = None
		return '{%s, %s, %s}' % (self.merge_type, self.split_type, children_str)
	
	def virtual_forward(self, x, init=False):
		return self.forward(x, virtual=True, init=init)
	
	def claim_ready(self, nBatch, noise=None):
		idx = 0
		for edge, child in zip(self.edges, self.child_nodes):
			branch_bn = None if self.branch_bns is None else self.branch_bns[idx]
			if edge:
				edge.claim_ready(nBatch, noise)
			if child:
				child.claim_ready(nBatch, noise)
			if branch_bn:
				branch_bn.running_mean /= nBatch
				branch_bn.running_var /= nBatch
				branch_bn.bias.data = branch_bn.running_mean.clone()
				branch_bn.weight.data = torch.sqrt(branch_bn.running_var + branch_bn.eps)
			idx += 1
	
	# -------------------------------- transformation operations -------------------------------- #
	
	def set_merge_type(self, merge_type, branch_num, noise=None):
		assert self.merge_type is None, 'current merge type is not None'
		assert self.child_num == 1 and self.child_nodes[0] is None, 'not applicable'
		
		edge = self.edges[0]
		self.merge_type = merge_type
		if merge_type == 'concat':
			split_list = self.get_split_list(edge.out_channels, branch_num)
			if isinstance(edge, IdentityLayer) or isinstance(edge, PoolingLayer):
				self.split_type = 'split'
			elif isinstance(edge, ConvLayer) and edge.groups > 1:
				self.split_type = 'split'
			else:
				self.split_type = 'copy'
			seg_edges = edge.split(split_list, noise)
			self.edges = nn.ModuleList(seg_edges)
		elif merge_type == 'add':
			self.split_type = 'copy'
			copy_edges = [edge.copy()] + [edge.copy(noise) for _ in range(branch_num - 1)]
			self.edges = nn.ModuleList(copy_edges)
		self.child_nodes = nn.ModuleList([None for _ in range(branch_num)])
		
	def insert_node(self, branch_idx):
		assert branch_idx < self.child_num, 'index out of range: %d' % branch_idx
		branch_edge = self.edges[branch_idx]
		branch_node = self.child_nodes[branch_idx]
		identity_edge = IdentityLayer(branch_edge.out_channels, branch_edge.out_channels, use_bn=False, act_func=None,
		                              dropout_rate=0, ops_order=branch_edge.ops_order)
		inserted_node = TreeNode(child_nodes=[branch_node], edges=[identity_edge], in_channels=branch_edge.out_channels,
		                         out_channels=branch_edge.out_channels, split_type=None, merge_type=None,
		                         use_avg=self.use_avg, bn_before_add=self.bn_before_add,
		                         path_drop_rate=self.path_drop_rate, use_zero_drop=self.use_zero_drop,
		                         drop_only_add=self.drop_only_add)
		self.child_nodes[branch_idx] = inserted_node
		
	def replace_identity_edge(self, idx, edge_config):
		assert idx < self.child_num, 'index out of range: %d' % idx
		old_edge = self.edges[idx]
		assert isinstance(old_edge, IdentityLayer), 'not applicable'
		
		edge_config['in_channels'] = old_edge.in_channels
		edge_config['out_channels'] = old_edge.out_channels
		edge_config['layer_ready'] = False
		
		if 'groups' in edge_config:
			groups = edge_config['groups']
			in_channels = edge_config['in_channels']
			while in_channels % groups != 0:
				groups -= 1
			edge_config['groups'] = groups
		new_edge = set_layer_from_config(edge_config)
		self.edges[idx] = new_edge



