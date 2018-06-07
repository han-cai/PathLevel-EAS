"""
Input: 'path' to the folder that contains configs and weights of a network
	Follow the configs and train the network
Output: 'Results' after training
"""

import json
import os
import hashlib

import torch

from models.run_manager import RunConfig, get_model_by_name, RunManger


def hash_str2int(text: str, algo='md5'):
	if algo == 'md5':
		val = hashlib.md5(text.encode('utf-8')).hexdigest()
	elif algo == 'sha1':
		val = hashlib.sha1(text.encode('utf-8')).hexdigest()
	elif algo == 'sha224':
		val = hashlib.sha224(text.encode('utf-8')).hexdigest()
	elif algo == 'sha256':
		val = hashlib.sha256(text.encode('utf-8')).hexdigest()
	elif algo == 'sha384':
		val = hashlib.sha384(text.encode('utf-8')).hexdigest()
	elif algo == 'sha512':
		val = hashlib.sha512(text.encode('utf-8')).hexdigest()
	else:
		raise NotImplementedError
	return int(val, 16)


class ExpdirMonitor:
	def __init__(self, expdir):
		self.expdir = os.path.realpath(expdir)
		os.makedirs(self.expdir, exist_ok=True)
	
	@property
	def logs_path(self):
		return '%s/logs' % self.expdir
	
	@property
	def save_path(self):
		return '%s/checkpoint' % self.expdir
	
	@property
	def output_path(self):
		return '%s/output' % self.expdir
	
	@property
	def init_path(self):
		return '%s/init' % self.expdir
	
	@property
	def run_config_path(self):
		return '%s/run.config' % self.expdir
	
	@property
	def net_config_path(self):
		return '%s/net.config' % self.expdir
	
	@property
	def net_str_path(self):
		return '%s/net.str' % self.expdir
	
	@property
	def trans_op_path(self):
		return '%s/trans.ops' % self.expdir
	
	@staticmethod
	def prepare_folder_for_valid(net_str, model, run_config, exp_dir, trans_ops, mimic_run_config=None, no_init=False):
		os.makedirs(exp_dir, exist_ok=True)
		monitor = ExpdirMonitor(exp_dir)
		json.dump(model.get_config(), open(monitor.net_config_path, 'w'), indent=4)
		json.dump(run_config.get_config(), open(monitor.run_config_path, 'w'), indent=4)
		if not no_init:
			torch.save({
				'state_dict': model.state_dict(),
			}, monitor.init_path)
		else:
			mimic_run_config = None
		
		# dump net_str and transformation ops
		json.dump({'net_str': net_str}, open(monitor.net_str_path, 'w'), indent=4)
		json.dump(trans_ops, open(monitor.trans_op_path, 'w'), indent=4)
		
		if mimic_run_config:
			json.dump(mimic_run_config, open('%s/mimic_run' % monitor.expdir, 'w'), indent=4)
	
	def load_output_val(self):
		if os.path.isfile(self.output_path):
			out_val = json.load(open(self.output_path, 'r'))
			if 'valid_acc' in out_val:
				return float(out_val['valid_acc'])
			elif 'test_acc' in out_val:
				return float(out_val['test_acc'])
		return 0.0
	
	def load_run_config(self, print_info=False, dataset='C10+'):
		if os.path.isfile(self.run_config_path):
			run_config = json.load(open(self.run_config_path, 'r'))
		else:
			print('Use Default Run Config for %s' % dataset)
			run_config = RunConfig.get_default_run_config(dataset)
		run_config = RunConfig(**run_config)
		if print_info:
			print('Run config:')
			for k, v in run_config.get_config().items():
				print('\t%s: %s' % (k, v))
		return run_config
	
	def load_model(self, print_info=False):
		assert os.path.isfile(self.net_config_path), 'No net configs found in <%s>' % self.expdir
		net_config_json = json.load(open(self.net_config_path, 'r'))
		if print_info:
			print('Net config:')
			for k, v in net_config_json.items():
				if k != 'blocks':
					print('\t%s: %s' % (k, v))
		model = get_model_by_name(net_config_json['name']).set_from_config(net_config_json)
		
		return model
	
	def load_init(self):
		if os.path.isfile(self.init_path):
			if torch.cuda.is_available():
				checkpoint = torch.load(self.init_path)
			else:
				checkpoint = torch.load(self.init_path, map_location='cpu')
			return checkpoint
		else:
			return None
		
	def run(self, pure=True, train=True, use_test_loader=True, valid_size=None, resume=False):
		init = self.load_init()
		dataset = 'C10+' if init is None else init.get('dataset', 'C10+')
		run_config = self.load_run_config(print_info=(not pure), dataset=dataset)
		run_config.renew_logs = False
		if valid_size is not None:
			run_config.valid_size = valid_size
		
		model = self.load_model(print_info=(not pure))
		run_manager = RunManger(self.expdir, model, run_config, out_log=(not pure), resume=resume)
		run_manager.save_config(print_info=(not pure))
		
		if not resume and init is not None:
			model_state_dict = run_manager.model.state_dict()
			model_key = list(model_state_dict.keys())[0]
			init_key = list(init['state_dict'].keys())[0]
			if model_key.startswith('module.'):
				if not init_key.startswith('module.'):
					new_state_dict = {}
					for key in init['state_dict']:
						new_state_dict['module.' + key] = init['state_dict'][key]
					init['state_dict'] = new_state_dict
			else:
				if init_key.startswith('module.'):
					new_state_dict = {}
					for key in init['state_dict']:
						new_key = '.'.join(key.split('.')[1:])
						new_state_dict[new_key] = init['state_dict'][key]
					init['state_dict'] = new_state_dict
			model_state_dict.update(init['state_dict'])
			run_manager.model.load_state_dict(model_state_dict)
		
		if pure:
			run_manager.pure_train()
			run_manager.save_model()
		elif train:
			run_manager.train()
			run_manager.save_model()
		
		if use_test_loader:
			loss, acc = run_manager.validate(use_test_loader=True)
			test_log = 'test_loss: %f\t test_acc: %f' % (loss, acc)
			if not pure:
				run_manager.write_log(test_log, prefix='test')
			json.dump({'test_loss': '%f' % loss, 'test_acc': '%f' % acc}, open(self.output_path, 'w'))
		else:
			loss, acc = run_manager.validate(use_test_loader=False)
			valid_log = 'valid_loss: %f\t valid_acc: %f' % (loss, acc)
			if not pure:
				run_manager.write_log(valid_log, prefix='valid')
			json.dump({'valid_loss': loss, 'valid_acc': acc, 'valid_size': run_config.valid_size},
			          open(self.output_path, 'w'))
		return acc
		
	def mimic_run(self, src_model_dir, sample_size=1000, distill_epochs=30, distill_lr=0.02):
		src_monitor = ExpdirMonitor(src_model_dir)
		src_init = src_monitor.load_init()
		src_model = src_monitor.load_model(print_info=False)
		if src_init is not None:
			model_state_dict = src_model.state_dict()
			model_state_dict.update(src_init['state_dict'])
			src_model.load_state_dict(model_state_dict)
		
		init = self.load_init()
		dataset = 'C10+' if init is None else init.get('dataset', 'C10+')
		run_config = self.load_run_config(print_info=False, dataset=dataset)
		run_config.renew_logs = False
		model = self.load_model(print_info=False)
		if init is not None:
			model_state_dict = model.state_dict()
			model_state_dict.update(init['state_dict'])
			model.load_state_dict(model_state_dict)
		
		train_loader, valid_loader, test_loader = run_config.build_data_provider()
		
		if torch.cuda.is_available():
			model = model.cuda()
			src_model = src_model.cuda()
		valid_loader.batch_size = 100
		
		model.mimic_run_with_linear_regression(valid_loader, src_model, sample_size=sample_size,
		                                       distill_epochs=distill_epochs, distill_lr=distill_lr, print_info=False)
		
		torch.save({
			'state_dict': model.state_dict(),
		}, self.init_path)

