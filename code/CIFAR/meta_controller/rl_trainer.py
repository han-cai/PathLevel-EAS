import numpy as np
import copy

import torch
import torch.optim as optim
from torch.autograd import Variable

from meta_controller.rl_controller import RLController


def get_rl_trainer_by_name(name: str):
	if name == ReinforceTrainer.__name__:
		return ReinforceTrainer
	else:
		raise NotImplementedError


class RLTrainer:
	def __init__(self, controller_net: RLController, entropy_penalty, opt_config):
		self.controller_net = controller_net
		self.entropy_penalty = entropy_penalty
		self.optimizer = self.build_optimizer(**opt_config)
	
	def build_optimizer(self, learning_rate, opt_name, opt_param):
		if opt_name == 'adam':
			return optim.Adam(self.controller_net.parameters(), lr=learning_rate, **opt_param)
		elif opt_name == 'sgd':
			return optim.SGD(self.controller_net.parameters(), lr=learning_rate, **opt_param)
		else:
			raise ValueError('Do not support the optimizer type: %s' % opt_name)
	
	def _get_gradient_data(self):
		gradient_dict = {}
		for name, param in self.controller_net.named_parameters():
			if param.grad is None:
				gradient_dict[name] = None  # torch.zeros(param.data.size())
			else:
				gradient_dict[name] = copy.deepcopy(param.grad.data.numpy())  # numpy ndarray
			if np.min(gradient_dict[name]) == 0 and np.max(gradient_dict[name]) == 0:
				gradient_dict[name] = None
		return gradient_dict
	
	def take_out_gradient(self, loss_term, retain_graph=False):
		self.controller_net.zero_grad()  # clear the gradient buffer
		loss_term.backward(retain_graph=retain_graph)  # compute gradient
		return self._get_gradient_data()  # take out gradient data
	
	######################################################################################
	
	@staticmethod
	def actor_entropy(probs):
		# flat probs -> large actor_entropy, obj
		eps = 1e-8
		log_probs = torch.log(probs + eps)  # [1, action_dim]
		actor_entropy = - torch.sum(torch.mul(probs, log_probs), dim=1)  # [1]
		return actor_entropy
	
	def forward_controller_net(self, whole_tree, focus_node, action_type):
		raise NotImplementedError
	
	def unrewarded_obj(self, decision, probs):
		raise NotImplementedError
	
	def on_action_taken(self, decision, probs, retain_graph=False):
		raise NotImplementedError
	
	######################################################################################
	
	def assign_gradient_data(self, gradient_dict):
		for name, param in self.controller_net.named_parameters():
			data = gradient_dict.get(name)
			if data is None:
				continue
			if isinstance(data, np.ndarray):
				data = torch.FloatTensor(data)
			if param.grad is None:
				param.grad = Variable(data, volatile=True)
			else:
				param.grad.data.copy_(data)
	
	@staticmethod
	def add_gradient_dict(target_dict, another_dict):
		for name, grad_data in another_dict.items():
			if grad_data is not None:
				if target_dict.get(name) is None:
					target_dict[name] = copy.deepcopy(grad_data)
				else:
					target_dict[name] += grad_data
	
	@staticmethod
	def mul_gradient_dict(target_dict, mul_value):
		for name in target_dict:
			if target_dict[name] is not None:
				target_dict[name] *= mul_value
	
	def on_reward_received(self, reward, gradient_list):
		raise NotImplementedError
	
	def update_controller(self, rewarded_gradients):
		batch_size = len(rewarded_gradients)
		# prepare gradient dict
		self.controller_net.zero_grad()
		cumulative_gradient_dict = self._get_gradient_data()
		for rewarded_grad_dict in rewarded_gradients:
			self.add_gradient_dict(cumulative_gradient_dict, rewarded_grad_dict)
		
		self.mul_gradient_dict(cumulative_gradient_dict, 1.0 / batch_size)  # / batch_size
		# assign gradient dict
		self.assign_gradient_data(cumulative_gradient_dict)
		# gradient clipping
		_max_grad_norm = 40
		torch.nn.utils.clip_grad_norm(self.controller_net.parameters(), _max_grad_norm)
		# apply gradient
		self.optimizer.step()
		# save model
		self.controller_net.save()


class ReinforceTrainer(RLTrainer):
	def forward_controller_net(self, whole_tree, focus_node, action_type):
		decision, probs = self.controller_net(whole_tree, focus_node, action_type)
		return decision, probs
	
	def unrewarded_obj(self, decision, probs):
		log_probs = torch.log(probs[0][decision])  # [1]
		return log_probs
	
	def on_action_taken(self, decision, probs, retain_graph=False):
		if self.entropy_penalty > 0:
			entropy_loss = - self.actor_entropy(probs) * self.entropy_penalty
			entropy_grad_dict = self.take_out_gradient(entropy_loss, retain_graph=True)
		else:
			entropy_grad_dict = None
		
		unrewarded_loss = - self.unrewarded_obj(decision, probs)
		unrewarded_grad_dict = self.take_out_gradient(unrewarded_loss, retain_graph=retain_graph)
		return unrewarded_grad_dict, entropy_grad_dict  # bath are `dict`
	
	def on_reward_received(self, reward, gradient_list):
		# gradient_list: list<unrewarded_grad_dict, entropy_grad_dict>
		for _i in range(len(gradient_list)):
			unrewarded_grad_dict, entropy_grad_dict = gradient_list[_i]
			self.mul_gradient_dict(unrewarded_grad_dict, reward)
			if entropy_grad_dict is not None:
				self.add_gradient_dict(unrewarded_grad_dict, entropy_grad_dict)
			gradient_list[_i] = unrewarded_grad_dict
