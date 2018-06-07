from layers import *


class TreeCellA(nn.Module):
	def __init__(self, in_channels, out_channels, groups=1, use_avg=False,
	             bn_before_add=False, path_drop=0, drop_only_add=False, use_zero_drop=True,
	             bias=False):
		super(TreeCellA, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		depth_conv_type_name = DepthseparableConv.__name__
		conv_1x1_type_name = Conv.__name__
		
		root_branch_config = {'type': Conv.__name__, 'kernel_size': 3, 'groups': groups}
		
		# cell configurations
		b1_b1 = TreeNode(
			edge_configs=[
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
				{'type': depth_conv_type_name, 'kernel_size': 7, 'groups': 1, 'bias': bias},
			], child_nodes=[None] * 2, in_channels=out_channels, out_channels=out_channels,
			split_type='split', merge_type='concat',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		b1_b2 = TreeNode(
			edge_configs=[
				{'type': depth_conv_type_name, 'kernel_size': 3, 'groups': 1, 'bias': bias},
				{'type': depth_conv_type_name, 'kernel_size': 5, 'groups': 1, 'bias': bias},
			], child_nodes=[None] * 2, in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		b1 = TreeNode(
			edge_configs=[
				{'type': conv_1x1_type_name, 'kernel_size': 1, 'groups': 1, 'bias': bias},
				{'type': 'Pool', 'op': 'max', 'kernel_size': 3},
			], child_nodes=[b1_b1, b1_b2], in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		b2_b1 = TreeNode(
			edge_configs=[
				{'type': depth_conv_type_name, 'kernel_size': 5, 'groups': 1, 'bias': bias},
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
			], child_nodes=[None] * 3, in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		b2_b2 = TreeNode(
			edge_configs=[
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
				{'type': depth_conv_type_name, 'kernel_size': 3, 'groups': 1, 'bias': bias},
			], child_nodes=[None] * 2, in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		b2 = TreeNode(
			edge_configs=[
				{'type': depth_conv_type_name, 'kernel_size': 5, 'groups': 1, 'bias': bias},
				{'type': depth_conv_type_name, 'kernel_size': 3, 'groups': 1, 'bias': bias},
			], child_nodes=[b2_b1, b2_b2], in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		self.root = TreeNode(
			edge_configs=[root_branch_config.copy() for _ in range(2)], child_nodes=[b1, b2],
			in_channels=in_channels, out_channels=out_channels, split_type='copy', merge_type='add', use_avg=use_avg,
			path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop, bn_before_add=bn_before_add
		)
	
	def forward(self, x):
		return self.root(x)


class TreeCellB(nn.Module):
	def __init__(self, in_channels, out_channels, groups=1, use_avg=False,
	             bn_before_add=False, path_drop=0, drop_only_add=False, use_zero_drop=True,
	             bias=False):
		super(TreeCellB, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		depth_conv_type_name = DepthseparableConv.__name__
		conv_1x1_type_name = Conv.__name__
		
		root_branch_config = {'type': Conv.__name__, 'kernel_size': 3, 'groups': groups}

		# cell configurations
		
		root_b1_b1 = TreeNode(
			edge_configs=[
				{'type': depth_conv_type_name, 'kernel_size': 5, 'groups': 1, 'bias': bias},
			], child_nodes=[None], in_channels=out_channels, out_channels=out_channels, split_type='', merge_type='',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		root_b1_b2 = TreeNode(
			edge_configs=[
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
			], child_nodes=[None], in_channels=out_channels, out_channels=out_channels, split_type='', merge_type='',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		root_b1 = TreeNode(
			edge_configs=[
				{'type': conv_1x1_type_name, 'kernel_size': 1, 'groups': 1, 'bias': bias},
				{'type': depth_conv_type_name, 'kernel_size': 7, 'groups': 1, 'bias': bias},
			], child_nodes=[root_b1_b1, root_b1_b2], in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		root_b2_b1 = TreeNode(
			edge_configs=[
				{'type': depth_conv_type_name, 'kernel_size': 7, 'groups': 1, 'bias': bias},
				{'type': conv_1x1_type_name, 'kernel_size': 1, 'groups': 1, 'bias': bias},
			], child_nodes=[None] * 2, in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		root_b2_b2 = TreeNode(
			edge_configs=[
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
			], child_nodes=[None], in_channels=out_channels, out_channels=out_channels, split_type='', merge_type='',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		root_b2 = TreeNode(
			edge_configs=[
				{'type': 'Pool', 'op': 'avg', 'kernel_size': 3},
				{'type': depth_conv_type_name, 'kernel_size': 7, 'groups': 1, 'bias': bias},
			], child_nodes=[root_b2_b1, root_b2_b2], in_channels=out_channels, out_channels=out_channels,
			split_type='copy', merge_type='add',
			use_avg=use_avg, path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop
		)
		
		self.root = TreeNode(
			edge_configs=[root_branch_config.copy() for _ in range(2)], child_nodes=[root_b1, root_b2],
			in_channels=in_channels, out_channels=out_channels, split_type='copy', merge_type='add', use_avg=use_avg,
			path_drop=path_drop, drop_only_add=drop_only_add, use_zero_drop=use_zero_drop, bn_before_add=bn_before_add
		)
	
	def forward(self, x):
		return self.root(x)
