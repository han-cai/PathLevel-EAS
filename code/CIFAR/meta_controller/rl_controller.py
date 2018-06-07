import os
import json

from meta_controller.encoder_net import TreeEncoderNet
from meta_controller.actor_nets import *


class RLController(nn.Module):
	""" actor_nets, save, load"""
	def __init__(self, path, actor_dict):
		super(RLController, self).__init__()
		self.path = os.path.realpath(path)
		os.makedirs(self.path, exist_ok=True)
		
		# actor networks
		self.name2actor = OrderedDict()
		actor_nets = []
		for name, config in actor_dict.items():
			actor = get_actor_by_name(name)(**config)
			actor_nets.append(actor)
			self.name2actor[name] = len(actor_nets) - 1
		self.actor_nets = nn.ModuleList(actor_nets)
	
	@property
	def save_path(self):
		return '%s/model' % self.path
	
	@property
	def logs_path(self):
		return '%s/logs' % self.path
	
	def load(self):
		if os.path.isfile('%s/model.path' % self.path):
			model_path = json.load(open('%s/model.path' % self.path, 'r'))
			try:
				if torch.cuda.is_available():
					self.load_state_dict(torch.load(model_path))
				else:
					self.load_state_dict(torch.load(model_path, map_location='cpu'))
				print('Load model from %s' % model_path)
			except Exception:
				print('Failed to to load model from save path: %s' % model_path)
		else:
			print('No model files in ' + '%s/model.path' % self.path)
	
	def save(self, global_step=None):
		path = self.save_path
		if global_step is not None:
			path += '-%d.pkl' % global_step
		else:
			path += '.pkl'
		torch.save(self.state_dict(), path)
		json.dump(path, open('%s/model.path' % self.path, 'w'))
	
	def actor(self, name):
		idx = self.name2actor[name]
		return self.actor_nets[idx]
	
	def init_actor(self, scheme):
		for actor in self.actor_nets:
			actor.init_actor(scheme)
	
	def forward(self, *_input):
		raise NotImplementedError


class RLTreeController(RLController):
	def __init__(self, path, tree_encoder_config, actor_dict):
		super(RLTreeController, self).__init__(path, actor_dict)
		
		# tree encoder network
		self.tree_encoder = TreeEncoderNet(**tree_encoder_config)
	
	def forward(self, whole_tree, focus_node, action_type):
		assert action_type in self.name2actor, 'Unsupported action type: %s' % action_type
		
		# encode the network
		self.tree_encoder(whole_tree)
		# go through corresponding actor net
		decision, probs = self.actor(action_type)(focus_node)
		
		return decision, probs
	
	def known_decision_forward(self, whole_tree, focus_node, action_type, decision):
		assert action_type in self.name2actor, 'Unsupported action type: %s' % action_type
		
		self.tree_encoder(whole_tree)
		decision, probs = self.actor(action_type)(focus_node, decision)
		return decision, probs
