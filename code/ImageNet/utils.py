from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
from functools import reduce
import operator

from layers import LearnedGroupConv, CondensingLinear, CondensingConv, Conv, LearnedLinear

count_ops = 0
count_params = 0


def get_num_gen(gen):
	return sum(1 for _ in gen)


def is_pruned(layer):
	try:
		layer.mask
		return True
	except AttributeError:
		return False


def is_leaf(model):
	return get_num_gen(model.children()) == 0


def convert_model(model, args):
	for m in model._modules:
		child = model._modules[m]
		if child is None: continue
		
		if is_pruned(child):
			if isinstance(child, LearnedGroupConv):
				model._modules[m] = CondensingConv(child)
			elif isinstance(child, LearnedLinear):
				model._modules[m] = CondensingLinear(child.get_masked_linear(), 0.5)
			else:
				raise NotImplementedError
			del (child)
		elif is_leaf(child):
			pass
		else:
			convert_model(child, args)


def get_layer_info(layer):
	layer_str = str(layer)
	type_name = layer_str[:layer_str.find('(')].strip()
	return type_name


def get_layer_param(model):
	return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


def measure_layer(layer, x):
	global count_ops, count_params
	multi_add = 1
	type_name = get_layer_info(layer)
	
	### ops_conv
	if type_name in ['Conv2d']:
		out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
		            layer.stride[0] + 1)
		out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
		            layer.stride[1] + 1)
		delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
		            layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
		delta_params = get_layer_param(layer)
		
	### ops_learned_conv
	elif type_name in ['LearnedGroupConv']:
		if not layer.only_conv:
			measure_layer(layer.relu, x)
			measure_layer(layer.norm, x)
		conv = layer.conv
		out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
		            conv.stride[0] + 1)
		out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
		            conv.stride[1] + 1)
		delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
		            conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
		delta_params = get_layer_param(conv) / layer.condense_factor
	
	### ops_linear
	elif type_name in ['Linear']:
		weight_ops = layer.weight.numel() * multi_add
		delta_ops = weight_ops
		delta_params = get_layer_param(layer)
		
	elif type_name in ['LearnedLinear']:
		weight_ops = layer.linear.weight.numel() * multi_add
		delta_ops = weight_ops * (1 - layer.drop_rate)
		delta_params = get_layer_param(layer) * (1 - layer.drop_rate)
	
	### ops_nothing
	elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'ReLU', 'AvgPool2d', 'MaxPool2d',
	                   'AdaptiveAvgPool2d']:
		delta_ops = 0
		delta_params = get_layer_param(layer)
		
	elif type_name in ['Identity']:
		delta_ops = 0
		delta_params = 0
		
	### unknown layer type
	else:
		raise TypeError('unknown layer type: %s' % type_name)
	
	count_ops += delta_ops
	count_params += delta_params
	return


def measure_model(model, H, W):
	global count_ops, count_params
	count_ops = 0
	count_params = 0
	data = Variable(torch.zeros(1, 3, H, W))
	
	def should_measure(x):
		return is_leaf(x) or is_pruned(x)
	
	def modify_forward(model):
		for child in model.children():
			if should_measure(child):
				def new_forward(m):
					def lambda_forward(x):
						measure_layer(m, x)
						return m.old_forward(x)
					
					return lambda_forward
				
				child.old_forward = child.forward
				child.forward = new_forward(child)
			else:
				modify_forward(child)
	
	def restore_forward(model):
		for child in model.children():
			# leaf node
			if should_measure(child) and hasattr(child, 'old_forward'):
				child.forward = child.old_forward
				child.old_forward = None
			else:
				restore_forward(child)
	
	modify_forward(model)
	model.forward(data)
	restore_forward(model)
	
	return count_ops, count_params

