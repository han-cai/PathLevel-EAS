import copy
from queue import Queue
import math

import torch.nn as nn

from models.layers import set_layer_from_config, ConvLayer, PoolingLayer, IdentityLayer, DepthConvLayer, LinearLayer
from models.tree_node import TreeNode
from models.utils import *


def get_block_by_name(name):
	if name == TransitionBlock.__name__:
		return TransitionBlock
	elif name == ChainBlock.__name__:
		return ChainBlock
	else:
		raise NotImplementedError


class ChainBlock(nn.Module):
	def __init__(self, cell):
		super(ChainBlock, self).__init__()
		
		self.cell = cell

	def forward(self, x):
		return self.cell(x)
	
	def get_config(self):
		return {
			'name': ChainBlock.__name__,
			'cell': self.cell.get_config(),
		}
	
	@staticmethod
	def set_from_config(config):
		cell = TreeNode.set_from_config(config.get('cell'))
		
		return ChainBlock(cell)
	
	def virtual_forward(self, x, init=False):
		x = self.cell.virtual_forward(x, init)
		
		return x
		
	def claim_ready(self, nBatch, noise=None):
		self.cell.claim_ready(nBatch, noise)
	

class ChainNet(BasicBlockWiseConvNet):
	def __init__(self, blocks, classifier, ops_order, tree_node_config):
		super(ChainNet, self).__init__(blocks, classifier)
		
		self.ops_order = ops_order
		self.tree_node_config = tree_node_config
	
	@property
	def building_block(self):
		for block in self.blocks:
			if isinstance(block, ChainBlock):
				return block.cell
		
	def get_config(self):
		return {
			'name': ChainNet.__name__,
			'ops_order': self.ops_order,
			'tree_node_config': self.tree_node_config,
			'blocks': [
				block.get_config() for block in self.blocks
			],
			'classifier': self.classifier.get_config(),
		}
	
	@staticmethod
	def set_from_config(config):
		blocks = []
		for block_config in config.get('blocks'):
			block = get_block_by_name(block_config.get('name'))
			tree_node_config = copy.deepcopy(config.get('tree_node_config'))
			if block == ChainBlock:
				block_config['cell'].update(tree_node_config)
				tree_node_config['bn_before_add'] = False
				tree_node_config['cell_drop_rate'] = 0
				to_updates = Queue()
				for child_config in block_config['cell']['child_nodes']:
					to_updates.put(child_config)
				while not to_updates.empty():
					child_config = to_updates.get()
					if child_config is not None:
						child_config.update(tree_node_config)
						for new_config in child_config['child_nodes']:
							to_updates.put(new_config)
			block = block.set_from_config(block_config)
			blocks.append(block)
		
		classifier_config = config.get('classifier')
		classifier = set_layer_from_config(classifier_config)
		
		ops_order = config.get('ops_order')
		
		return ChainNet(blocks, classifier, ops_order, config.get('tree_node_config'))
	
	@staticmethod
	def set_standard_net(data_shape, n_classes, start_planes, block_per_group, total_groups, downsample_type='avg_pool',
	                     ops_order='bn_act_weight', dropout_rate=0,
	                     use_identity=False, use_depth_sep_conv=False, groups_3x3=1,
	                     path_drop_rate=0, use_zero_drop=True, drop_only_add=False):
		image_channel, image_size = data_shape[0:2]
		
		# initial conv
		features_dim = start_planes
		if ops_order == 'weight_bn_act':
			init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=True, act_func='relu',
			                            dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'act_weight_bn':
			init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=True, act_func=None,
			                            dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'bn_act_weight':
			init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=False, act_func=None,
			                            dropout_rate=0, ops_order=ops_order)
		else:
			raise NotImplementedError
		transition2blocks = TransitionBlock([init_conv_layer])
		blocks = [transition2blocks]
		
		if use_identity:
			assert 'pool' in downsample_type, 'Error'
		for group_idx in range(total_groups):
			for block_idx in range(block_per_group):
				if downsample_type is None and group_idx > 0 and block_idx == 0:
					stride = 2
					image_size //= 2
					out_dim = features_dim * 2
				else:
					stride = 1
					out_dim = features_dim
				# prepare the chain block
				if use_identity:
					cell_edge = IdentityLayer(features_dim, features_dim, use_bn=False, act_func=None,
					                         dropout_rate=0, ops_order=ops_order)
				elif use_depth_sep_conv:
					cell_edge = DepthConvLayer(features_dim, out_dim, kernel_size=3, stride=stride, use_bn=True,
					                           act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					cell_edge = ConvLayer(features_dim, out_dim, kernel_size=3, stride=stride, groups=groups_3x3,
					                      use_bn=True, act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				cell = TreeNode(child_nodes=[None], edges=[cell_edge], in_channels=features_dim,
				                out_channels=out_dim, split_type=None, merge_type=None,
				                path_drop_rate=path_drop_rate, use_zero_drop=use_zero_drop,
				                drop_only_add=drop_only_add)
				chain_block = ChainBlock(cell)
				blocks.append(chain_block)
				features_dim = out_dim
			if group_idx != total_groups - 1 and downsample_type is not None:
				if 'pool' in downsample_type:
					conv_layer = ConvLayer(features_dim, 2 * features_dim, kernel_size=1, stride=1, use_bn=True,
					                       act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
					features_dim *= 2
					if downsample_type == 'avg_pool':
						pool_layer = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=2, stride=2,
						                          use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
					else:
						pool_layer = PoolingLayer(features_dim, features_dim, 'max', kernel_size=2, stride=2,
						                          use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
					transition_layers = [conv_layer, pool_layer]
				elif downsample_type == 'conv':
					conv_layer = ConvLayer(features_dim, 2 * features_dim, kernel_size=3, stride=2, use_bn=True,
					                       act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
					features_dim *= 2
					transition_layers = [conv_layer]
				else:
					raise NotImplementedError
				image_size //= 2
				inter_block_transition = TransitionBlock(transition_layers)
				blocks.append(inter_block_transition)
		
		if ops_order == 'weight_bn_act':
			global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
			                               use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'act_weight_bn':
			global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
			                               use_bn=False, act_func='relu', dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'bn_act_weight':
			global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
			                               use_bn=True, act_func='relu', dropout_rate=0, ops_order=ops_order)
		else:
			raise NotImplementedError
		transition2classes = TransitionBlock([global_avg_pool])
		blocks.append(transition2classes)
		
		classifier = LinearLayer(features_dim, n_classes, bias=True)
		tree_node_config = {
			'use_avg': True,
			'bn_before_add': False,
			'path_drop_rate': path_drop_rate,
			'use_zero_drop': use_zero_drop,
			'drop_only_add': drop_only_add,
		}
		
		return ChainNet(blocks, classifier, ops_order, tree_node_config)



