"""
Manage the folder for architecture search
"""
import os
import json
from collections import deque
import numpy as np
import copy
from sys import stderr

from expdir_monitor.expdir_monitor import ExpdirMonitor
from expdir_monitor import distributed
from expdir_monitor.net_pool import NetPool
from meta_controller.rl_trainer import RLTrainer


class ArchManager:
	def __init__(self, arch_path, exp_settings, batch_size=10, reward_config=None, random=False):
		# gpu cluster
		if os.path.isfile(distributed.config_file):
			config_path = distributed.config_file
		else:
			config_path = '../%s' % distributed.config_file
		self.gpu_cluster = distributed.ClusterController(json.load(open(config_path, 'r')))
		
		self.task_queue = deque()
		self.running_queue = deque()
		self.result_slots = {}
		self.unrewarded_buffer = {}
		self.update_queue = deque()
		
		# folder for the meta_controller and logs of architecture search
		self.arch_path = os.path.realpath(arch_path)
		os.makedirs(self.arch_path, exist_ok=True)
		
		self.batch_size = batch_size
		self.reward_config = reward_config
		self.random = random
		self.trainer = None  # to be set
		
		# load start nets and prepare net_pools
		self.start_nets = list()
		self.net_pools = list()
		start_nets_val = []
		for start_net_path, net_pool_path in exp_settings:
			start_net_monitor = ExpdirMonitor(start_net_path)
			print('=====================================')
			print('Load start net from %s' % start_net_monitor.expdir)
			init = start_net_monitor.load_init()
			dataset = 'C10+' if init is None else init.get('dataset', 'C10+')
			run_config = start_net_monitor.load_run_config(print_info=True, dataset=dataset)
			run_config.renew_logs = False
			
			model = start_net_monitor.load_model(print_info=True)
			if init is not None:
				loaded_state_dict = init['state_dict']
				model_state_dict = model.state_dict()
				model_state_dict.update(loaded_state_dict)
				model.load_state_dict(model_state_dict)
			
			data_provider = run_config.build_data_provider()
			self.start_nets.append([model, run_config, data_provider])
			
			net_pool = NetPool(net_pool_path)
			self.net_pools.append(net_pool)
			
			start_nets_val.append(start_net_monitor.load_output_val())
		
		self.trained_nets = 0
		self.val_wrt_nets = list()
		self.val_wrt_nets.append(np.array(start_nets_val))
		
		self.val_logger = open(self.val_logs_path, 'a')  # performance information of sampled nets
		self.net_logger = open(self.net_logs_path, 'a')  # architecture information of sampled net
		
		self.baseline = self.get_reward(start_nets_val)  # baseline function
		
		# load existing logs, update <self.trained_nets> and <self.val_wrt_nets>
		with open(self.val_logs_path, 'r') as fin:
			reward_list = []
			for line in fin.readlines():
				line = line[:-1].split('\t')
				is_new = int(line[2].split('=')[-1])
				net_val_list = line[3].split('=')[-1].split('_')
				net_val_list = [float(val) for val in net_val_list]
				
				self.trained_nets += is_new
				self.val_wrt_nets.append(np.array(net_val_list))
				# update baseline function
				reward_list.append(self.get_reward(net_val_list))
				if len(reward_list) % batch_size == 0:
					self.update_baseline(np.mean(reward_list))
					reward_list.clear()
	
	@property
	def meta_controller_path(self):
		return '%s/controller' % self.arch_path
	
	@property
	def val_logs_path(self):
		return '%s/val.log' % self.arch_path
	
	@property
	def net_logs_path(self):
		return '%s/net.log' % self.arch_path
	
	@property
	def exp_setting_num(self):
		return len(self.net_pools)
	
	#################################################################################################
	
	@property
	def rolling_avg_val_wrt_nets(self):
		""" average of last <batch_size> sampled nets """
		avg_val = []
		for net_val in reversed(self.val_wrt_nets[1:]):
			avg_val.append(net_val)
			if len(avg_val) >= self.batch_size: break
		avg_val = np.array(avg_val)
		return np.mean(avg_val, axis=0)
	
	@property
	def max_val_wrt_nets(self):
		return np.array(self.val_wrt_nets[1:]).max(axis=0)
	
	def max_reward(self):
		max_net_val = np.array(self.val_wrt_nets).max(axis=0)
		max_reward = self.get_reward(max_net_val)
		return max_reward
	
	def log_net(self, net_str, results, is_new=True, print_info=True):
		""" write to files and also update related variables hosted in Arch-Manager, i.e. trained_nets, val_wrt_nets """
		if is_new: self.trained_nets += 1  # increase <trained_nets> if it is_new
		self.val_wrt_nets.append(np.array(results))  # update <val_wrt_nets>
		sampled_nets = len(self.val_wrt_nets) - 1
		arch_manager_status = 'sampled=%d\t trained=%d\t is_new=%d' % (sampled_nets, self.trained_nets, is_new)
		
		# write architecture information (net_str) of the sampled net via <net_logger>
		net_log_str = arch_manager_status + '\t' + net_str
		self.net_logger.write(net_log_str + '\n')
		self.net_logger.flush()
		
		# write validation performance information via <val_logger>
		net_val_str = '_'.join([str(val) for val in results])
		avg_val_str = '_'.join([str(val) for val in self.rolling_avg_val_wrt_nets])
		max_val_str = '_'.join([str(val) for val in self.max_val_wrt_nets])
		val_log = '%s\t current=%s\t rolling_mean=%s\t max=%s' % \
		          (arch_manager_status, net_val_str, avg_val_str, max_val_str)
		if print_info: print(val_log)
		self.val_logger.write(val_log + '\n')
		self.val_logger.flush()
	
	#################################################################################################
	
	def get_reward(self, results):
		""" reward_config: {'func': XX, 'merge': XX, 'decay': XX, 'div_std': XX} """
		transformed_res = []
		trans_func = self.reward_config.get('func', None)
		for val in results:
			if val > 1:
				assert val < 100, 'Invalid value'
				val /= 100
			if trans_func is None:
				transformed_res.append(val)
			elif trans_func == 'tan':
				transformed_res.append(np.tan(val * np.pi / 2))
			else:
				raise NotImplementedError
		merge_mode = self.reward_config.get('merge', None)
		if merge_mode is None:
			reward = np.mean(transformed_res)
		else:
			raise NotImplementedError
		return reward
	
	def update_baseline(self, mean_reward):
		# exp moving average
		decay = self.reward_config['decay']
		self.baseline += decay * (mean_reward - self.baseline)
	
	def sample_exp_settings(self):
		return [setting_idx for setting_idx in range(self.exp_setting_num)]
	
	def get_start_net(self, setting_idx=0, _copy=False):
		""" given the exp_setting idx, return the corresponding start network """
		assert setting_idx < self.exp_setting_num, 'Index out of range'
		
		net_config, run_config, data_provider = self.start_nets[setting_idx]
		if _copy:
			return copy.deepcopy(net_config), copy.deepcopy(run_config), data_provider
		else:
			return net_config, run_config, data_provider
	
	def get_net_val(self, net_str, setting_idx=0):
		assert setting_idx < self.exp_setting_num, 'Index out of range'
		
		net_val, net_folder = self.net_pools[setting_idx].get_net_val(net_str)
		return net_val, net_folder
	
	#################################################################################################
	
	def on_net_val_received(self, net_str, setting_idx, net_val):
		self.net_pools[setting_idx].add_net(net_str, net_val, save=True)  # add to corresponding net pool
		
		slot_dict = self.result_slots[net_str]
		slot_dict[setting_idx] = net_val  # fill the corresponding result slot
		if np.all([isinstance(val, float) for val in slot_dict.values()]):  # check if slots filled
			# if yes, trigger _on_result_slots_filled
			self.on_result_slots_filled(net_str, is_new=True)
	
	def on_result_slots_filled(self, net_str, is_new=True):
		results = [0.0] * self.exp_setting_num
		slot_dict = self.result_slots.pop(net_str)
		for setting_idx, net_val in slot_dict.items():
			# take out the net from <results_slots>
			results[setting_idx] = net_val
		
		# add logs, update <trained_nets>, <val_wrt_nets>
		self.log_net(net_str, results, is_new=is_new)
		# add to update_queue
		self.add2update_queue(results, net_str)
	
	def add2update_queue(self, results, net_str):
		reward = self.get_reward(results)  # calculate the reward
		# take out from unrewarded buffer & add to update queue
		
		for unrewarded_gradients in self.unrewarded_buffer.pop(net_str):
			self.update_queue.appendleft([reward, unrewarded_gradients])
		if len(self.update_queue) >= self.batch_size:
			self.update_meta_controller()
	
	def update_meta_controller(self):
		assert isinstance(self.trainer, RLTrainer), 'must set the trainer before update the meta controller'
		
		reward_list = []
		to_reward_gradients = []
		for _i in range(self.batch_size):
			reward, unrewarded_grads = self.update_queue.pop()
			reward_list.append(reward)
			to_reward_gradients.append(unrewarded_grads)
		
		if self.random: return
		
		self.update_baseline(np.mean(reward_list))  # update baseline function
		reward_list = np.array(reward_list) - self.baseline
		
		if np.std(reward_list) == 0:
			print('No need to perform this update with all cells being the same: %s' % reward_list, file=stderr)
			return
		
		if self.reward_config.get('div_std', None):
			reward_list /= self.max_reward() - self.baseline
		# reward_list /= np.std(reward_list)
		
		# update the meta controller
		for _i in range(self.batch_size):
			reward = reward_list[_i]
			to_reward_grad = to_reward_gradients[_i]
			self.trainer.on_reward_received(reward, to_reward_grad)
			# sum of gradients
			for _j in range(1, len(to_reward_grad)):
				self.trainer.add_gradient_dict(to_reward_grad[0], to_reward_grad[_j])
			to_reward_gradients[_i] = to_reward_grad[0]
		
		self.trainer.update_controller(to_reward_gradients)
		print('meta controller updated')



