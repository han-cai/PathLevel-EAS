import math
from queue import Queue
import copy

from models.layers import ConvLayer, PoolingLayer, IdentityLayer, DepthConvLayer, LinearLayer
from models.tree_node import TreeNode
from models.utils import *


def get_block_by_name(name):
	if name == TransitionBlock.__name__:
		return TransitionBlock
	elif name == ResidualBlock.__name__:
		return ResidualBlock
	else:
		raise NotImplementedError
	

class ResidualBlock(nn.Module):
	def __init__(self, cell, in_bottle, out_bottle, shortcut, final_bn=False):
		super(ResidualBlock, self).__init__()
		
		self.cell = cell
		self.in_bottle = in_bottle
		self.out_bottle = out_bottle
		self.shortcut = shortcut
		
		if final_bn:
			if self.out_bottle is None:
				out_channels = self.cell.out_channels
			else:
				out_channels = self.out_bottle.out_channels
			self.final_bn = nn.BatchNorm2d(out_channels)
		else:
			self.final_bn = None
	
	def forward(self, x):
		_x = self.shortcut(x)
		
		if self.in_bottle is not None:
			x = self.in_bottle(x)
		
		x = self.cell(x)
		
		if self.out_bottle is not None:
			x = self.out_bottle(x)
		if self.final_bn:
			x = self.final_bn(x)
		
		residual_channel = x.size()[1]
		shortcut_channel = _x.size()[1]
		
		batch_size = x.size()[0]
		featuremap = x.size()[2:4]
		if residual_channel != shortcut_channel:
			padding = torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap[0], featuremap[1])
			if x.is_cuda:
				padding = padding.cuda()
			padding = torch.autograd.Variable(padding)
			_x = torch.cat((_x, padding), 1)
		
		return _x + x
	
	def get_config(self):
		return {
			'name': ResidualBlock.__name__,
			'shortcut': self.shortcut.get_config(),
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
			
		shortcut = set_layer_from_config(config.get('shortcut'))
		cell = TreeNode.set_from_config(config.get('cell'))
		final_bn = config.get('final_bn', False)
		
		return ResidualBlock(cell, in_bottle, out_bottle, shortcut, final_bn)
	
	def virtual_forward(self, x, init=False):
		_x = self.shortcut.virtual_forward(x, init)
		
		if self.in_bottle is not None:
			x = self.in_bottle.virtual_forward(x, init)
		x = self.cell.virtual_forward(x, init)
		if self.out_bottle is not None:
			x = self.out_bottle.virtual_forward(x, init)
		if self.final_bn:
			x = self.final_bn(x)
		
		residual_channel = x.size()[1]
		shortcut_channel = _x.size()[1]
		
		batch_size = x.size()[0]
		featuremap = x.size()[2:4]
		if residual_channel != shortcut_channel:
			padding = torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap[0], featuremap[1])
			if x.is_cuda:
				padding = padding.cuda()
			padding = torch.autograd.Variable(padding)
			_x = torch.cat((_x, padding), 1)
		
		return _x + x
	
	def claim_ready(self, nBatch, noise=None):
		if self.in_bottle:
			self.in_bottle.claim_ready(nBatch, noise)
		self.cell.claim_ready(nBatch, noise)
		if self.out_bottle:
			self.out_bottle.claim_ready(nBatch, noise)
		self.shortcut.claim_ready(nBatch, noise)
	
	
class PyramidNet(BasicBlockWiseConvNet):
	def __init__(self, blocks, classifier, ops_order, tree_node_config, groups_3x3):
		super(PyramidNet, self).__init__(blocks, classifier)
		
		self.ops_order = ops_order
		self.tree_node_config = tree_node_config
		self.groups_3x3 = groups_3x3
	
	@property
	def building_block(self):
		for block in self.blocks:
			if isinstance(block, ResidualBlock):
				return block.cell
	
	def get_config(self):
		return {
			'name': PyramidNet.__name__,
			'ops_order': self.ops_order,
			'tree_node_config': self.tree_node_config,
			'groups_3x3': self.groups_3x3,
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
			if block == ResidualBlock:
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
		groups_3x3 = config.get('groups_3x3', 1)

		return PyramidNet(blocks, classifier, ops_order, config.get('tree_node_config'), groups_3x3)
	
	@staticmethod
	def set_standard_net(data_shape, n_classes, start_planes, alpha, block_per_group, total_groups, downsample_type,
	                     bottleneck=4, ops_order='bn_act_weight', dropout_rate=0,
	                     final_bn=True, no_first_relu=True, use_depth_sep_conv=False, groups_3x3=1,
	                     path_drop_rate=0, use_zero_drop=True, drop_only_add=False):
		image_channel, image_size = data_shape[0:2]
		
		addrate = alpha / (block_per_group * total_groups)  # add pyramid_net
		
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
		if final_bn:
			init_bn_layer = IdentityLayer(features_dim, features_dim, use_bn=True, act_func=None,
			                              dropout_rate=0, ops_order=ops_order)
			transition2blocks = TransitionBlock([init_conv_layer, init_bn_layer])
		else:
			transition2blocks = TransitionBlock([init_conv_layer])
		blocks = [transition2blocks]
		
		planes = start_planes
		for group_idx in range(total_groups):
			for block_idx in range(block_per_group):
				if group_idx > 0 and block_idx == 0:
					stride = 2
					image_size //= 2
				else:
					stride = 1
				# prepare the residual block
				planes += addrate
				if stride == 1:
					shortcut = IdentityLayer(features_dim, features_dim, use_bn=False, act_func=None,
					                         dropout_rate=0, ops_order=ops_order)
				else:
					if downsample_type == 'avg_pool':
						shortcut = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=2, stride=2,
						                        use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
					elif downsample_type == 'max_pool':
						shortcut = PoolingLayer(features_dim, features_dim, 'max', kernel_size=2, stride=2,
						                        use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
					else:
						raise NotImplementedError
				
				out_plane = int(round(planes))
				if out_plane % groups_3x3 != 0:
					out_plane -= out_plane % groups_3x3  # may change to +=
				if no_first_relu:
					in_bottle = ConvLayer(features_dim, out_plane, kernel_size=1, use_bn=True, act_func=None,
					                      dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					in_bottle = ConvLayer(features_dim, out_plane, kernel_size=1, use_bn=True, act_func='relu',
					                      dropout_rate=dropout_rate, ops_order=ops_order)
				
				if use_depth_sep_conv:
					cell_edge = DepthConvLayer(out_plane, out_plane, kernel_size=3, stride=stride, use_bn=True,
					                           act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					cell_edge = ConvLayer(out_plane, out_plane, kernel_size=3, stride=stride, groups=groups_3x3,
					                      use_bn=True, act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				cell = TreeNode(child_nodes=[None], edges=[cell_edge], in_channels=out_plane,
				                out_channels=out_plane, split_type=None, merge_type=None,
				                path_drop_rate=path_drop_rate, use_zero_drop=use_zero_drop,
				                drop_only_add=drop_only_add)
				
				out_bottle = ConvLayer(out_plane, out_plane * bottleneck, kernel_size=1, use_bn=True,
				                       act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				residual_block = ResidualBlock(cell, in_bottle, out_bottle, shortcut, final_bn=final_bn)
				blocks.append(residual_block)
				features_dim = out_plane * bottleneck
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
		
		return PyramidNet(blocks, classifier, ops_order, tree_node_config, groups_3x3)

