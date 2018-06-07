import math
from queue import Queue
import copy

from models.layers import ConvLayer, PoolingLayer, IdentityLayer, DepthConvLayer, LinearLayer
from models.tree_node import TreeNode
from models.utils import *


def get_block_by_name(name):
	if name == TransitionBlock.__name__:
		return TransitionBlock
	elif name == DenseBlock.__name__:
		return DenseBlock


class DenseBlock(nn.Module):
	def __init__(self, cell, in_bottle, out_bottle, final_bn=False):
		super(DenseBlock, self).__init__()
		self.cell = cell
		self.in_bottle = in_bottle
		self.out_bottle = out_bottle
		if final_bn:
			if self.out_bottle is None:
				out_channels = self.cell.out_channels
			else:
				out_channels = self.out_bottle.out_channels
			self.final_bn = nn.BatchNorm2d(out_channels)
		else:
			self.final_bn = None
		
	def forward(self, x):
		x_ = x
		if self.in_bottle:
			x = self.in_bottle(x)
		
		x = self.cell(x)
		
		if self.out_bottle:
			x = self.out_bottle(x)
		if self.final_bn:
			x = self.final_bn(x)
			
		return torch.cat([x_, x], 1)
	
	def get_config(self):
		return {
			'name': DenseBlock.__name__,
			'in_bottle': None if self.in_bottle is None else self.in_bottle.get_config(),
			'out_bottle': None if self.out_bottle is None else self.out_bottle.get_config(),
			'final_bn': False if self.final_bn is None else True,
			'cell': self.cell.get_config(),
		}
	
	@staticmethod
	def set_from_config(config):
		if config.get('in_bottle'):
			in_bottle = set_layer_from_config(config.get('in_bottle'))
		else:
			in_bottle = None
		
		if config.get('out_bottle'):
			out_bottle = set_layer_from_config(config.get('out_bottle'))
		else:
			out_bottle = None
		
		final_bn = config.get('final_bn', False)
		cell = TreeNode.set_from_config(config.get('cell'))
		
		return DenseBlock(cell, in_bottle, out_bottle, final_bn)
	
	def virtual_forward(self, x, init=False):
		x_ = x
		if self.in_bottle:
			x = self.in_bottle.virtual_forward(x, init)
		x = self.cell.virtual_forward(x, init)
		if self.out_bottle:
			x = self.out_bottle.virtual_forward(x, init)
		if self.final_bn:
			x = self.final_bn(x)
		return torch.cat([x_, x], 1)
	
	def claim_ready(self, nBatch, noise=None):
		if self.in_bottle:
			self.in_bottle.claim_ready(nBatch, noise)
		self.cell.claim_ready(nBatch, noise)
		if self.out_bottle:
			self.out_bottle.claim_ready(nBatch, noise)
			
			
class DenseNet(BasicBlockWiseConvNet):
	def __init__(self, blocks, classifier, ops_order, tree_node_config):
		super(DenseNet, self).__init__(blocks, classifier)
		
		self.ops_order = ops_order
		self.tree_node_config = tree_node_config
	
	@property
	def building_block(self):
		for block in self.blocks:
			if isinstance(block, DenseBlock):
				return block.cell
	
	def get_config(self):
		return {
			'name': DenseNet.__name__,
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
			if block == DenseBlock:
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
		
		return DenseNet(blocks, classifier, ops_order, config.get('tree_node_config'))
	
	@staticmethod
	def set_standard_net(data_shape, n_classes, growth_rate, dense_block_per_group, total_groups, downsample_type,
	                     first_ratio=2, reduction=0.5, bottleneck=4,
	                     final_bn=False, no_first_relu=False, use_depth_sep_conv=False, groups_3x3=1,
	                     ops_order='bn_act_weight', dropout_rate=0,
	                     path_drop_rate=0, use_zero_drop=True, drop_only_add=False):
		image_channel, image_size = data_shape[0:2]
		features_dim = growth_rate * first_ratio
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
		if final_bn:
			init_bn_layer = IdentityLayer(features_dim, features_dim, use_bn=True, act_func=None,
			                              dropout_rate=0, ops_order=ops_order)
			transition2blocks = TransitionBlock([init_conv_layer, init_bn_layer])
		else:
			transition2blocks = TransitionBlock([init_conv_layer])
		
		blocks = [transition2blocks]
		for group_idx in range(total_groups):
			for block_idx in range(dense_block_per_group):
				if no_first_relu:
					in_bottle = ConvLayer(features_dim, growth_rate * bottleneck, kernel_size=1, use_bn=True,
					                      act_func=None, dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					in_bottle = ConvLayer(features_dim, growth_rate * bottleneck, kernel_size=1, use_bn=True,
					                      act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				
				if use_depth_sep_conv:
					cell_edge = DepthConvLayer(growth_rate * bottleneck, growth_rate, kernel_size=3, use_bn=True,
					                           act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					cell_edge = ConvLayer(growth_rate * bottleneck, growth_rate, kernel_size=3, groups=groups_3x3,
					                      use_bn=True, act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				cell = TreeNode(child_nodes=[None], edges=[cell_edge], in_channels=growth_rate * bottleneck,
				                out_channels=growth_rate, split_type=None, merge_type=None,
				                path_drop_rate=path_drop_rate, use_zero_drop=use_zero_drop, drop_only_add=drop_only_add)
				dense_block = DenseBlock(cell, in_bottle, out_bottle=None, final_bn=final_bn)
				blocks.append(dense_block)
				features_dim += growth_rate
			if group_idx != total_groups - 1:
				if downsample_type == 'pool':
					pool_layer = PoolingLayer(features_dim, features_dim, 'avg',
					                          kernel_size=2, stride=2, use_bn=False, act_func=None, dropout_rate=0,
					                          ops_order=ops_order)
					transition_layers = [pool_layer]
				elif downsample_type == 'conv-pool':
					conv_layer = ConvLayer(features_dim, int(features_dim * reduction),
					                       kernel_size=1, stride=1, use_bn=True, act_func='relu',
					                       dropout_rate=dropout_rate, ops_order=ops_order)
					features_dim = int(features_dim * reduction)
					pool_layer = PoolingLayer(features_dim, features_dim, 'avg',
					                          kernel_size=2, stride=2, use_bn=False, act_func=None, dropout_rate=0,
					                          ops_order=ops_order)
					transition_layers = [conv_layer, pool_layer]
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
		
		return DenseNet(blocks, classifier, ops_order, tree_node_config)


