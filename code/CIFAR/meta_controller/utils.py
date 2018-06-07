from meta_controller.actor_nets import *
from models.layers import *
from models.tree_node import TreeNode


class Vocabulary:
	def __init__(self, token_list):
		self.token_list = [self.pad_token()] + token_list
		self.vocab = OrderedDict()
		for idx, token in enumerate(self.token_list):
			self.vocab[token] = idx
	
	@property
	def size(self):
		return len(self.token_list)
	
	@property
	def pad_code(self):
		return self.vocab[self.pad_token()]
	
	@staticmethod
	def pad_token():
		return '#PAD'
	
	def get_code(self, in_token):
		if isinstance(in_token, list):
			return [self.vocab[token] for token in in_token]
		else:
			assert isinstance(in_token, str), 'Invalid'
			return self.vocab[in_token]
	
	def get_token(self, in_code):
		if isinstance(in_code, list):
			return [self.token_list[code] for code in in_code]
		else:
			assert isinstance(in_code, int), 'Invalid'
			return self.token_list[in_code]


class Tree:
	def __init__(self, parent, idx, edge, children, _type):
		self.parent = parent  # `Tree`
		self.idx = idx
		self.edge = edge  # `str`, the edge that connects the tree to its parent
		
		self.type = _type  # `str`
		self.children = children
		
		# encoder results
		self.state = None
	
	def __eq__(self, other_tree):
		if self.idx != other_tree.idx: return False
		if self.edge != other_tree.edge: return False
		if self.type != other_tree.type: return False
		if self.child_num != other_tree.child_num: return False
		
		for self_child, other_child in zip(self.children, other_tree.children):
			if self_child != other_child: return False
		
		return True
	
	@property
	def child_num(self):
		return len(self.children)
	
	@property
	def is_leaf(self):
		return self.child_num == 0
	
	@property
	def is_root(self):
		return self.parent is None
	
	@property
	def depth(self):
		if self.parent is None:
			return 0
		else:
			return 1 + self.parent.depth
		
	@property
	def height(self):
		if self.is_leaf:
			return 0
		else:
			return np.max([child.height for child in self.children]) + 1
	
	@property
	def tree_str(self):
		if self.child_num > 0:
			child_str = []
			for _i, child in enumerate(self.children):
				child_str.append('%s=>%s' % (child.edge, child.tree_str))
			child_str = '[%s]' % ', '.join(child_str)
		else:
			child_str = None
		return '{%s, %s}' % (self.type, child_str)
	
	@property
	def root_node(self):
		node = self
		while node.parent is not None:
			node = node.parent
		return node
	
	@property
	def get_path_from_root(self):
		if self.parent is None:
			return []
		else:
			parent_path = self.parent.get_path_from_root
			return parent_path + [self.idx]
	
	def get_node(self, node_path):
		node = self.root_node
		for branch in node_path:
			node = node.children[branch]
		return node
	
	@staticmethod
	def build_tree_from_torch_module(root_node: TreeNode):
		# parent, idx, edge; children, _type
		if root_node is None:
			return Tree(parent=None, idx=None, edge=None, children=list(), _type=None)
		
		if root_node.merge_type is not None:
			_type = '%s-%s' % (root_node.merge_type, root_node.split_type)
		else:
			_type = None
		tree = Tree(parent=None, idx=None, edge=None, children=list(), _type=_type)
		for _i, child_node in enumerate(root_node.child_nodes):
			child_tree = Tree.build_tree_from_torch_module(child_node)
			# set child: parent, idx, edge
			child_tree.parent = tree
			child_tree.idx = _i
			child_tree.edge = root_node.edges[_i].get_str
			tree.children.append(child_tree)
		return tree
		
	# adjust the tree according to the transformation operation
	def apply_trans_op(self, _type, op_param):
		# parent, idx, edge; children, _type
		if _type == TreeNode.SET_MERGE_TYPE:
			assert self.child_num == 1 and self.children[0].is_leaf, 'Invalid'
			
			merge_type, branch_num = op_param['merge_type'], op_param['branch_num']
			# change node type
			child_edge = self.children[0].edge
			if merge_type == 'concat' and child_edge in ['Identity', '3x3_GroupConv', Vocabulary.pad_token()]:
				sep_type = 'split'
			else:
				sep_type = 'copy'
			self.type = '%s-%s' % (merge_type, sep_type)
			# update children
			self.children.clear()
			for _i in range(branch_num):
				new_child_edge = child_edge
				if new_child_edge == 'Identity': new_child_edge = Vocabulary.pad_token()
				
				new_child = Tree(parent=self, idx=_i, edge=new_child_edge, children=list(), _type=None)
				self.children.append(new_child)
		elif _type == TreeNode.INSERT_NODE:
			child_idx = op_param['branch_idx']
			original_child_node = self.children[child_idx]
			
			new_node = Tree(parent=self, idx=child_idx, edge=original_child_node.edge, children=list(), _type=None)
			
			original_child_node.parent = new_node
			original_child_node.idx = 0
			original_child_node.edge = Vocabulary.pad_token()
			
			new_node.children.append(original_child_node)
			self.children[child_idx] = new_node
		elif _type == TreeNode.REPLACE_IDENTITY_EDGE:
			idx, edge_type = op_param['idx'], op_param.pop('edge_type')
			self.children[idx].edge = edge_type
		else:
			raise NotImplementedError
	
	def get_state(self):
		if self.state is not None:
			bottom_up_state, top_down_state = self.state
			if top_down_state is None or self.parent is None:
				return bottom_up_state  # [1, hidden_size]
			else:
				h = torch.cat([bottom_up_state[0], top_down_state[0]], dim=1)  # [1, 2 * hidden_size]
				c = torch.cat([bottom_up_state[1], top_down_state[1]], dim=1)  # [1, 2 * hidden_size]
				return h, c
		else:
			return None
	
	def get_output(self):
		state = self.get_state()
		if state is None:
			return None
		else:
			return state[0]
	
	def clear_state(self):
		self.state = None
		for child in self.children:
			child.clear_state()
