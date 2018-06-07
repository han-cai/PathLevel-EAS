from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_actor_by_name(name):
	if name == SetMergeTypeActor.__name__:
		return SetMergeTypeActor
	elif name == InsertNodeActor.__name__:
		return InsertNodeActor
	elif name == ReplaceIdentityEdgeActor.__name__:
		return ReplaceIdentityEdgeActor
	else:
		raise NotImplementedError


def build_linear_seq(in_dim, out_dim, config_list):
	layer_seq = OrderedDict()
	dim = in_dim
	for _i, layer_config in enumerate(config_list):
		layer_seq['linear_%d' % _i] = nn.Linear(dim, layer_config['units'], bias=layer_config['use_bias'])
		if layer_config['activation'] == 'sigmoid':
			layer_seq['sigmoid_%d' % _i] = nn.Sigmoid()
		elif layer_config['activation'] == 'tanh':
			layer_seq['tanh_%d' % _i] = nn.Tanh()
		elif layer_config['activation'] == 'relu':
			layer_seq['relu_%d' % _i] = nn.ReLU(inplace=True)
		dim = layer_config['units']
	layer_seq['final_linear'] = nn.Linear(dim, out_dim)
	return nn.Sequential(layer_seq)


def init_linear_seq(linear_seq: nn.Sequential, scheme: dict):
	for module in linear_seq.modules():
		if isinstance(module, nn.Linear):
			if module.bias is not None:
				module.bias.data.zero_()
			_type = scheme.get('type', 'default')
			if _type == 'uniform':
				_min, _max = scheme.get('min', -0.1), scheme.get('max', 0.1)
				module.weight.data.uniform_(_min, _max)
			elif _type == 'normal':
				_mean, _std = scheme.get('mean', 0), scheme.get('std', 0.1)
				module.weight.data.normal_(_mean, _std)
			elif _type == 'default':
				pass
			else:
				raise NotImplementedError


# decide merge type, also branch number
class SetMergeTypeActor(nn.Module):
	def __init__(self, merge_type_candidates, branch_num_candidates, encoder_hidden_size, actor_config):
		super(SetMergeTypeActor, self).__init__()
		
		self.merge_type_candidates = merge_type_candidates
		self.branch_num_candidates = branch_num_candidates
		self.encoder_hidden_size = encoder_hidden_size
		self.actor_config = actor_config
		
		# define parameters
		self.merge_type_decider = build_linear_seq(self.encoder_hidden_size, len(self.merge_type_candidates),
		                                           self.actor_config)
		branch_num_decider = []  # separate branch_num decider for each merge_type
		for idx in range(len(self.merge_type_candidates)):
			branch_num_list = self.branch_num_candidates[idx]
			if len(branch_num_list) <= 1:
				decider = None
			else:
				decider = build_linear_seq(self.encoder_hidden_size, len(branch_num_list), self.actor_config)
			branch_num_decider.append(decider)
		self.branch_num_decider = nn.ModuleList(branch_num_decider)
	
	def init_actor(self, scheme: dict):
		init_linear_seq(self.merge_type_decider, scheme)
		for decider in self.branch_num_decider:
			if decider is not None:
				init_linear_seq(decider, scheme)
	
	def forward(self, tree, decision=None):
		if decision:
			return self._known_decision_forward(tree, decision)
		
		state = tree.get_output()  # [1, hidden_size]
		merge_type_logits = self.merge_type_decider(state)  # [1, merge_type_size]
		merge_type_probs = F.softmax(merge_type_logits, dim=1)  # [1, merge_type_size]
		# sample a decision
		merge_type_decision = torch.multinomial(merge_type_probs, 1, replacement=True)  # [1, 1]
		merge_type_decision = merge_type_decision.data.numpy()[0, 0]  # int
		
		if self.branch_num_decider[merge_type_decision] is None:
			branch_num_probs = None
			branch_num_decision = None
		else:
			branch_num_logits = self.branch_num_decider[merge_type_decision](state)  # [1, branch_num_size]
			branch_num_probs = F.softmax(branch_num_logits, dim=1)  # [1, branch_num_size]
			# sample a decision
			branch_num_decision = torch.multinomial(branch_num_probs, 1, replacement=True)  # [1, 1]
			branch_num_decision = branch_num_decision.data.numpy()[0, 0]  # int
		
		return [merge_type_decision, branch_num_decision], [merge_type_probs, branch_num_probs]
	
	def _known_decision_forward(self, tree, decision):
		assert isinstance(decision, list)
		state = tree.get_output()  # [1, hidden_size]
		merge_type_logits = self.merge_type_decider(state)  # [1, merge_type_size]
		merge_type_probs = F.softmax(merge_type_logits, dim=1)  # [1, merge_type_size]
		
		merge_type_decision, _ = decision
		
		if self.branch_num_decider[merge_type_decision] is None:
			branch_num_probs = None
		else:
			branch_num_logits = self.branch_num_decider[merge_type_decision](state)  # [1, branch_num_size]
			branch_num_probs = F.softmax(branch_num_logits, dim=1)  # [1, branch_num_size]
		
		return decision, [merge_type_probs, branch_num_probs]

	def random_decision(self):
		merge_type_decision = np.random.randint(0, len(self.merge_type_candidates))
		if self.branch_num_decider[merge_type_decision] is None:
			branch_num_decision = None
		else:
			branch_num_decision = np.random.randint(0, len(self.branch_num_candidates[merge_type_decision]))
		return merge_type_decision, branch_num_decision
	

# decide whether to expand a leaf node, `sigmoid classifier`
class InsertNodeActor(nn.Module):
	def __init__(self, encoder_hidden_size, actor_config):
		super(InsertNodeActor, self).__init__()
		
		self.encoder_hidden_size = encoder_hidden_size
		self.actor_config = actor_config
		
		# define parameters
		self.expand_decider = build_linear_seq(self.encoder_hidden_size, 1, self.actor_config)
	
	def init_actor(self, scheme: dict):
		init_linear_seq(self.expand_decider, scheme)
	
	def forward(self, tree, decision=None):
		if decision:
			return self._known_decision_forward(tree, decision)
		
		state = tree.get_output()  # [1, hidden_size]
		expand_logits = self.expand_decider(state)  # [1, 1]
		expand_probs = F.sigmoid(expand_logits)  # [1, 1]
		expand_probs = torch.cat(
			[1 - expand_probs, expand_probs], dim=1,
		)  # [1, 2]
		
		# sample a decision
		expand_decision = torch.multinomial(expand_probs, 1, replacement=True)  # [1, 1]
		expand_decision = expand_decision.data.numpy()[0, 0]  # int
		
		return expand_decision, expand_probs
	
	def _known_decision_forward(self, tree, decision):
		_, probs = self.forward(tree)
		return decision, probs
	
	@staticmethod
	def random_decision():
		expand_decision = np.random.randint(0, 2)
		return expand_decision


# decide the edge type, `softmax classifier`
class ReplaceIdentityEdgeActor(nn.Module):
	def __init__(self, edge_candidates, encoder_hidden_size, actor_config):
		super(ReplaceIdentityEdgeActor, self).__init__()
		
		self.edge_candidates = edge_candidates
		self.encoder_hidden_size = encoder_hidden_size
		self.actor_config = actor_config
		
		# define parameters
		self.edge_decider = build_linear_seq(self.encoder_hidden_size, len(self.edge_candidates), self.actor_config)
	
	def init_actor(self, scheme: dict):
		init_linear_seq(self.edge_decider, scheme)
	
	def forward(self, tree, decision=None):
		if decision:
			return self._known_decision_forward(tree, decision)
		state = tree.get_output()  # [1, hidden_size]
		edge_logits = self.edge_decider(state)  # [1, edge_candidates_num]
		edge_probs = F.softmax(edge_logits, dim=1)  # [1, edge_candidates_num]
		
		# sample a decision
		edge_decision = torch.multinomial(edge_probs, 1, replacement=True)  # [1, 1]
		edge_decision = edge_decision.data.numpy()[0, 0]  # int
		
		return edge_decision, edge_probs
	
	def _known_decision_forward(self, tree, decision):
		_, probs = self.forward(tree)
		return decision, probs
	
	def random_decision(self):
		return np.random.randint(0, len(self.edge_candidates))
