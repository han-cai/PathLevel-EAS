import torch.nn.init as init
from torch.autograd import Variable

from meta_controller.utils import *


def lstm_zero_hidden_state(hidden_size):
	h0 = Variable(torch.zeros(1, hidden_size))
	c0 = Variable(torch.zeros(1, hidden_size))
	return h0, c0  # [1, hidden_size]


class BottomUpTreeLSTMCell(nn.Module):
	def __init__(self, input_dim, hidden_size, max_n=2, _type='child-sum&n-ary'):
		super(BottomUpTreeLSTMCell, self).__init__()
		self.type = _type.split('&')
		self.hidden_size = hidden_size
		self.max_n = max_n
		
		# define parameters
		self.iou_x = nn.Linear(input_dim, 3 * hidden_size)  # bias term here
		self.f_x = nn.Linear(input_dim, hidden_size)  # bias term here
		
		if 'child-sum' in self.type:
			# child sum
			self.iou_h_sum = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
			self.f_h_sum = nn.Linear(hidden_size, hidden_size, bias=False)
		if 'n-ary' in self.type:
			# n-ary
			self.iou_h_nArray = nn.Linear(max_n * hidden_size, 3 * hidden_size, bias=False)
			self.f_h_nArray = nn.ModuleList([
				nn.Linear(max_n * hidden_size, hidden_size, bias=False) for _ in range(max_n)
			])
	
	@property
	def rnn_parameter_list(self):
		# 5 + max_n <weights>, 2 <bias>
		weight_parameters = [self.iou_x.weight, self.f_x.weight]
		bias_parameters = [self.iou_x.bias, self.f_x.bias]
		
		if 'child-sum' in self.type:
			weight_parameters += [self.iou_h_sum.weight, self.f_h_sum.weight]
		if 'n-ary' in self.type:
			weight_parameters += [self.iou_h_nArray.weight] + [self.f_h_nArray[k].weight for k in range(self.max_n)]
		
		return weight_parameters, bias_parameters
	
	def forward(self, op_type, input_x, child_h: list, child_c: list):
		assert op_type in self.type, 'Unsupported operation: %s' % op_type
		child_num = len(child_c)
		
		child_h = torch.cat(child_h, dim=0)  # [child_num, hidden_size]
		child_c = torch.cat(child_c, dim=0)  # [child_num, hidden_size]
		
		if op_type == 'child-sum':
			child_sum_h = torch.sum(child_h, dim=0, keepdim=True)  # [1, hidden_size]
			iou = self.iou_x(input_x) + self.iou_h_sum(child_sum_h)  # [1, 3 * hidden_size]
			i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
			i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
			
			f = F.sigmoid(
				self.f_x(input_x).repeat(child_num, 1) + self.f_h_sum(child_h)
			)  # [child_num, hidden_size]
			fc = torch.mul(f, child_c)  # [child_num, hidden_size]
			
			c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)  # [1, hidden_size]
			h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
		elif op_type == 'n-ary':
			if child_num < self.max_n:
				padding = torch.zeros(self.max_n - child_num, self.hidden_size)
				child_cat_h = torch.cat([child_h, Variable(padding)], dim=0)  # [max_n, hidden_size]
			else:
				child_cat_h = child_h
			
			child_cat_h = child_cat_h.view(1, -1)  # [1, max_n * hidden_size]
			iou = self.iou_x(input_x) + self.iou_h_nArray(child_cat_h)  # [1, 3 * hidden_size]
			i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
			i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
			
			fk = [
				F.sigmoid(self.f_x(input_x) + self.f_h_nArray[k](child_cat_h)) for k in range(child_num)
			]  # each is [1, hidden_size]
			f = torch.cat(fk, dim=0)  # [child_num, hidden_size]
			fc = torch.mul(f, child_c)  # [child_num, hidden_size]
			
			c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)  # [1, hidden_size]
			h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
		else:
			raise NotImplementedError
		
		return h, c


class TopDownTreeLSTMCell(nn.Module):
	def __init__(self, input_dim, hidden_size, max_n=2, _type='child-sum&n-ary'):
		super(TopDownTreeLSTMCell, self).__init__()
		self.type = _type.split('&')
		self.hidden_size = hidden_size
		self.max_n = max_n
		
		# define parameters
		self.iou_x = nn.Linear(input_dim, 3 * hidden_size)  # bias term here
		self.f_x = nn.Linear(input_dim, hidden_size)  # bias term here
		
		if 'child-sum' in self.type:
			self.iou_h_sum_child = nn.Linear(hidden_size, 3 * hidden_size, bias=False)  # h_child_sum
			self.iou_h_sum_parent = nn.Linear(hidden_size, 3 * hidden_size, bias=False)  # h_parent
			
			self.f_h_sum_child = nn.Linear(2 * hidden_size, hidden_size, bias=False)  # [h_parent, h_k]
			self.f_h_sum_parent = nn.Linear(2 * hidden_size, hidden_size, bias=False)  # [h_parent, h_child_sum]
		if 'n-ary' in self.type:
			self.iou_h_nArray_child = nn.Linear(max_n * hidden_size, 3 * hidden_size, bias=False)
			self.iou_h_nArray_parent = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
			
			self.f_h_nArray_child = nn.ModuleList([
				nn.Linear((max_n + 1) * hidden_size, hidden_size, bias=False) for _ in range(max_n)
			])  # [h_parent, h_1, h_2, ..., h_N]
			self.f_h_nArray_parent = nn.Linear((max_n + 1) * hidden_size, hidden_size, bias=False)
	
	@property
	def rnn_parameter_list(self):
		# 9 + max_n <weight>, 2 <bias>
		weight_parameters = [self.iou_x.weight, self.f_x.weight]
		bias_parameters = [self.iou_x.bias, self.f_x.bias]
		
		if 'child-sum' in self.type:
			weight_parameters += [self.iou_h_sum_child.weight, self.iou_h_sum_parent.weight]
			weight_parameters += [self.f_h_sum_child.weight, self.f_h_sum_parent.weight]
		if 'n-ary' in self.type:
			weight_parameters += [self.iou_h_nArray_child.weight, self.iou_h_nArray_parent.weight]
			weight_parameters += [self.f_h_nArray_child[k].weight for k in range(self.max_n)] + \
			                     [self.f_h_nArray_parent.weight]
		
		return weight_parameters, bias_parameters
	
	def forward(self, op_type, input_x, parent_h, parent_c, child_h: list, child_c: list):
		assert op_type in self.type, 'Unsupported operation: %s' % op_type
		child_num = len(child_c)
		
		child_h = torch.cat(child_h, dim=0)  # [child_num, hidden_size]
		child_c = torch.cat(child_c, dim=0)  # [child_num, hidden_size]
		
		if op_type == 'child-sum':
			child_sum_h = torch.sum(child_h, dim=0, keepdim=True)  # [1, hidden_size], sum of all except <child_idx>
			iou = self.iou_x(input_x) + self.iou_h_sum_child(child_sum_h) + self.iou_h_sum_parent(parent_h)
			i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
			i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
			
			fk_child = [
				F.sigmoid(self.f_x(input_x) + self.f_h_sum_child(torch.cat([parent_h, child_h[k:k + 1]], dim=1)))
				for k in range(child_num)
			]  # each is [1, hidden_size]
			f_child = torch.cat(fk_child, dim=0)  # [child_num, hidden_size]
			fc_child = torch.mul(f_child, child_c)  # [child_num, hidden_size], <child_idx> row should be zeros
			
			f_parent = F.sigmoid(
				self.f_x(input_x) + self.f_h_sum_parent(torch.cat([parent_h, child_sum_h], dim=1))
			)  # [1, hidden_size]
			fc_parent = torch.mul(f_parent, parent_c)  # [1, hidden_size]
			
			c = torch.mul(i, u) + torch.sum(fc_child, dim=0, keepdim=True) + fc_parent  # [1, hidden_size]
			h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
		elif op_type == 'n-ary':
			if child_num < self.max_n:
				padding = torch.zeros(self.max_n - child_num, self.hidden_size)
				child_cat_h = torch.cat([child_h, Variable(padding)], dim=0)  # [max_n, hidden_size]
			else:
				child_cat_h = child_h
			child_cat_h = child_cat_h.view(1, -1)  # [1, max_n * hidden_size]
			iou = self.iou_x(input_x) + self.iou_h_nArray_child(child_cat_h) + self.iou_h_nArray_parent(parent_h)
			i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
			i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
			
			parent_child_cat_h = torch.cat([parent_h, child_cat_h], dim=1)  # [1, (max_n + 1) * hidden_size]
			fk_child = [
				F.sigmoid(self.f_x(input_x) + self.f_h_nArray_child[k](parent_child_cat_h)) for k in range(child_num)
			]  # each is [1, hidden_size]
			f_child = torch.cat(fk_child, dim=0)  # [child_num, hidden_size]
			fc_child = torch.mul(f_child, child_c)  # [child_num, hidden_size], <child_idx> row should be zeros
			
			f_parent = F.sigmoid(
				self.f_x(input_x) + self.f_h_nArray_parent(parent_child_cat_h)
			)  # [1, hidden_size]
			fc_parent = torch.mul(f_parent, parent_c)  # [1, hidden_size]
			
			c = torch.mul(i, u) + torch.sum(fc_child, dim=0, keepdim=True) + fc_parent  # [1, hidden_size]
			h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
		else:
			raise NotImplementedError
		
		return h, c


class TreeEncoderNet(nn.Module):
	def __init__(self, edge_vocab, node_vocab, max_n, embedding_dim, hidden_size, bidirectional=True):
		super(TreeEncoderNet, self).__init__()
		
		self.edge_vocab = edge_vocab
		self.node_vocab = node_vocab
		
		self.max_n = max_n  # max children number for N-ary case
		self.embedding_dim = embedding_dim
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		
		# embedding layer
		self.edge_embedding = nn.Embedding(self.edge_vocab.size, self.embedding_dim, self.edge_vocab.pad_code)
		self.node_embedding = nn.Embedding(self.node_vocab.size, self.embedding_dim, self.node_vocab.pad_code)
		
		# RNN parameters
		self.bottom_up_node_transition_lstm = \
			BottomUpTreeLSTMCell(self.embedding_dim, self.hidden_size, self.max_n, _type='child-sum&n-ary')
		self.bottom_up_edge_transition_lstm = nn.LSTMCell(self.embedding_dim, self.hidden_size)
		
		if self.bidirectional:
			self.top_down_node_transition_lstm = \
				TopDownTreeLSTMCell(self.embedding_dim, self.hidden_size, self.max_n, _type='child-sum&n-ary')
			self.top_down_edge_transition_lstm = nn.LSTMCell(self.embedding_dim, self.hidden_size)
	
	def init_embedding(self, scheme: dict):
		_type = scheme.get('type', 'uniform')
		if _type == 'uniform':
			_min = scheme.get('min', -0.1)
			_max = scheme.get('max', 0.1)
			init.uniform(self.edge_embedding.weight, _min, _max)
			init.uniform(self.node_embedding.weight, _min, _max)
		elif _type == 'normal':
			_mean = scheme.get('mean', 0)
			_std = scheme.get('std', 1)
			init.normal(self.edge_embedding.weight, _mean, _std)
			init.normal(self.node_embedding.weight, _mean, _std)
		elif _type == 'default':
			pass
		else:
			raise NotImplementedError
		if self.edge_embedding.padding_idx is not None:
			self.edge_embedding.weight.data[self.edge_embedding.padding_idx].fill_(0)
		if self.node_embedding.padding_idx is not None:
			self.node_embedding.weight.data[self.node_embedding.padding_idx].fill_(0)
	
	@property
	def rnn_parameter_list(self):
		weight_parameters, bias_parameters = self.bottom_up_node_transition_lstm.rnn_parameter_list
		
		weight_parameters += [
			self.bottom_up_edge_transition_lstm.weight_hh, self.bottom_up_edge_transition_lstm.weight_ih
		]
		if self.bottom_up_edge_transition_lstm.bias:
			bias_parameters += [
				self.bottom_up_edge_transition_lstm.bias_hh, self.bottom_up_edge_transition_lstm.bias_ih
			]
		
		if self.bidirectional:
			top_down_weight_parameters, top_down_bias_parameters = self.top_down_node_transition_lstm.rnn_parameter_list
			weight_parameters += top_down_weight_parameters
			bias_parameters += top_down_bias_parameters
			weight_parameters += [
				self.top_down_edge_transition_lstm.weight_hh, self.top_down_edge_transition_lstm.weight_ih
			]
			if self.top_down_edge_transition_lstm.bias:
				bias_parameters += [
					self.top_down_edge_transition_lstm.bias_hh, self.top_down_edge_transition_lstm.bias_ih
				]
		return weight_parameters, bias_parameters
	
	def init_cell(self, scheme: dict):
		_type = scheme.get('type', 'default')
		weight_parameters, bias_parameters = self.rnn_parameter_list
		if _type == 'uniform':
			_min, _max = scheme.get('min', -0.1), scheme.get('max', 0.1)
			for weight in weight_parameters:
				init.uniform(weight, _min, _max)
			for bias in bias_parameters:
				init.uniform(bias, _min, _max)
		elif _type == 'normal':
			_mean, _std = scheme.get('mean', 0), scheme.get('std', 1)
			for weight in weight_parameters:
				init.normal(weight, _mean, _std)
			for bias in bias_parameters:
				init.normal(bias, _mean, _std)
		elif _type == 'orthogonal':
			for weight in weight_parameters:
				init.orthogonal(weight)
			for bias in bias_parameters:
				init.constant(bias, 0.0)
		elif _type == 'default':
			pass
		else:
			raise NotImplementedError
	
	########################################################################################################
	
	def _zero_hidden_state(self):
		return lstm_zero_hidden_state(self.hidden_size)
	
	def _bottom_up(self, tree: Tree):
		if tree.is_leaf:
			child_h, child_c = self._zero_hidden_state()  # [1, hidden_size]
			child_h, child_c = [child_h], [child_c]
		else:
			child_h, child_c = [], []
			for child in tree.children:
				self._bottom_up(child)
				bottom_up_state = child.state[0]  # (h, c)
				child_h.append(bottom_up_state[0])
				child_c.append(bottom_up_state[1])
		
		if tree.type is None:
			# no need to perform node transition
			assert len(child_h) == len(child_c) == 1
			op_type = None
		elif tree.type.startswith('add'):
			op_type = 'child-sum'
		elif tree.type.startswith('concat'):
			op_type = 'n-ary'
		else:
			raise NotImplementedError
		if op_type is None:
			h, c = child_h[0], child_c[0]  # [1, hidden_size]
		else:
			# perform node transition
			node_type_idx = self.node_vocab.get_code(tree.type)
			input_node = Variable(torch.LongTensor([node_type_idx]))
			input_node = self.node_embedding(input_node)  # [1, embedding_dim]
			h, c = self.bottom_up_node_transition_lstm(op_type, input_node, child_h, child_c)  # [1, hidden_size]
		
		if tree.edge is None:
			# no need to perform edge transition
			pass
		else:
			# perform edge transition
			edge_idx = self.edge_vocab.get_code(tree.edge)  # int
			input_edge = Variable(torch.LongTensor([edge_idx]))
			input_edge = self.edge_embedding(input_edge)  # [1, embedding_dim]
			h, c = self.bottom_up_edge_transition_lstm(input_edge, (h, c))  # [1, hidden_size]
		
		tree.state = [(h, c), None]
	
	def _top_down(self, tree: Tree):
		if tree.parent is None:
			top_down_state = self._zero_hidden_state()  # h: [1, hidden_size], c: [1, hidden_size]
		else:
			parent = tree.parent
			assert isinstance(parent, Tree), 'Invalid'
			parent_h, parent_c = parent.state[1]  # top_down_state of parent
			
			child_h, child_c = [], []
			for child in parent.children:
				if child.idx == tree.idx:
					child_bottom_up_state = lstm_zero_hidden_state(self.hidden_size)
				else:
					child_bottom_up_state = child.state[0]
				child_h.append(child_bottom_up_state[0])
				child_c.append(child_bottom_up_state[1])
			
			if parent.type is None:
				# no need to perform node transition
				assert len(child_h) == len(child_c) == 1 and tree.idx == 0, 'Invalid'
				op_type = None
			elif parent.type.startswith('add'):
				op_type = 'child-sum'
			elif parent.type.startswith('concat'):
				op_type = 'n-ary'
			else:
				raise NotImplementedError
			if op_type is None:
				h, c = parent_h, parent_c
			else:
				# perform node transition
				node_type_idx = self.node_vocab.get_code(parent.type)
				input_node = Variable(torch.LongTensor([node_type_idx]))
				input_node = self.node_embedding(input_node)  # [1, embedding_dim]
				h, c = self.top_down_node_transition_lstm(op_type, input_node, parent_h, parent_c, child_h, child_c)
			
			if tree.edge is None:
				# no need to perform edge transition
				top_down_state = (h, c)
			else:
				# perform edge transition
				edge_idx = self.edge_vocab.get_code(tree.edge)  # int
				input_edge = Variable(torch.LongTensor([edge_idx]))
				input_edge = self.edge_embedding(input_edge)  # [1, embedding_dim]
				top_down_state = self.top_down_edge_transition_lstm(input_edge, (h, c))  # [1, hidden_size]
		
		tree.state[1] = top_down_state
		for child in tree.children:
			self._top_down(child)
	
	def forward(self, tree: Tree):
		self._bottom_up(tree)
		
		if self.bidirectional:
			self._top_down(tree)
		return tree.get_output()
