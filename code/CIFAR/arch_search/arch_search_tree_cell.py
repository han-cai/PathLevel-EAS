from time import sleep
from collections import deque
import json

from expdir_monitor.arch_manager import ArchManager
from meta_controller.utils import *
from meta_controller.actor_nets import *
from meta_controller.rl_controller import RLTreeController
from meta_controller.rl_trainer import *
from models.tree_node import TreeNode
from models.densenet import DenseBlock
from models.pyramidnet import ResidualBlock
from models.chain_net import ChainBlock
from expdir_monitor.expdir_monitor import ExpdirMonitor
from expdir_monitor.net_pool import hash_str2int


def virtual_run(exp_dir):
	monitor = ExpdirMonitor(exp_dir)
	net_str = json.load(open(monitor.net_str_path, 'r'))['net_str']
	algos = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512']
	virtual_val = [(hash_str2int(net_str, algo=algo) % 400 - 200) / 100.0 for algo in algos]
	virtual_val = [1 / (1 + np.exp(-val)) for val in virtual_val]
	virtual_val = np.mean(virtual_val)
	virtual_val = float((0.8 * virtual_val + 1.2 * np.tanh(len(net_str) / 150))) / 2
	json.dump({'valid_acc': '%f' % virtual_val}, open(monitor.output_path, 'w'))
	return virtual_val


def sample_cell(start_cell, rl_trainer, net_config, noise_config, random=False):
	cell_space_config = {
		'max_depth': 3,
		'depth_first': True  # else width_first
	}
	
	#########################################################################################################
	
	trans_ops_record = list()
	unrewarded_gradients = list()
	whole_tree = Tree.build_tree_from_torch_module(start_cell)
	for child_tree in whole_tree.children:
		if child_tree.edge == 'Identity':
			child_tree.edge = Vocabulary.pad_token()
	
	node_queue = deque()
	node_queue.append([SetMergeTypeActor.__name__, whole_tree.children[0]])
	
	while len(node_queue) > 0:
		trans_op_type, focus_node = node_queue.pop()
		whole_tree.clear_state()  # empty state buffer
		decision, probs = rl_trainer.forward_controller_net(whole_tree, focus_node, trans_op_type)  # forward
		
		if trans_op_type == SetMergeTypeActor.__name__:
			merge_type_decision, branch_num_decision = decision
			merge_type_probs, branch_num_probs = probs
			
			# take out unrewarded_gradients, backward
			if branch_num_decision is not None:
				branch_num_gradients_dict = rl_trainer.on_action_taken(branch_num_decision, branch_num_probs, True)
				unrewarded_gradients.append(branch_num_gradients_dict)
			merge_type_gradient_dict = rl_trainer.on_action_taken(merge_type_decision, merge_type_probs, False)
			unrewarded_gradients.append(merge_type_gradient_dict)
			
			# decode trans_op
			corresponding_actor = rl_trainer.controller_net.actor(trans_op_type)
			
			if random:
				merge_type_decision, branch_num_decision = corresponding_actor.random_decision()
			
			merge_type = corresponding_actor.merge_type_candidates[merge_type_decision]
			
			branch_num_list = corresponding_actor.branch_num_candidates[merge_type_decision]
			# apply trans op
			parent_node = focus_node.parent
			if branch_num_decision is None: branch_num_decision = 0
			if merge_type == 'none':
				pass
			else:
				branch_num = branch_num_list[branch_num_decision]
				op_param = {
					'merge_type': merge_type,
					'branch_num': branch_num,
					'noise': noise_config,
				}
				trans_op = [TreeNode.SET_MERGE_TYPE, parent_node.get_path_from_root, op_param]
				trans_ops_record.append(trans_op)  # add to trans_ops_record
				
				parent_node.apply_trans_op(TreeNode.SET_MERGE_TYPE, op_param)
			
			# update <node_queue>
			if cell_space_config['depth_first']:
				# depth_first
				for child_node in reversed(parent_node.children):
					if child_node.depth < cell_space_config['max_depth']:
						node_queue.append([InsertNodeActor.__name__, child_node])
			else:
				# width_first
				for child_node in parent_node.children:
					if child_node.depth < cell_space_config['max_depth']:
						node_queue.appendleft([InsertNodeActor.__name__, child_node])
			# set edges, the first priority after set_merge_type
			for child_node in reversed(parent_node.children):
				if child_node.edge == Vocabulary.pad_token():
					node_queue.append([ReplaceIdentityEdgeActor.__name__, child_node])
		elif trans_op_type == InsertNodeActor.__name__:
			expand_decision, expand_probs = decision, probs
			
			# take out unrewarded_gradients, backward
			expand_gradient_dict = rl_trainer.on_action_taken(expand_decision, expand_probs, False)
			unrewarded_gradients.append(expand_gradient_dict)
			
			if random:
				expand_decision = InsertNodeActor.random_decision()
			# expand_decision == 1 means expand the node, else not
			parent_node = focus_node.parent
			if expand_decision:
				op_param = {
					'branch_idx': focus_node.idx,
				}
				trans_op = [TreeNode.INSERT_NODE, parent_node.get_path_from_root, op_param]
				trans_ops_record.append(trans_op)
				
				parent_node.apply_trans_op(TreeNode.INSERT_NODE, op_param)
				
				# update <node_queue>
				next_focus_node = parent_node.children[op_param['branch_idx']].children[0]
				if cell_space_config['depth_first']:
					# depth_first
					node_queue.append([SetMergeTypeActor.__name__, next_focus_node])
				else:
					# width_first
					node_queue.appendleft([SetMergeTypeActor.__name__, next_focus_node])
			else:
				pass
		elif trans_op_type == ReplaceIdentityEdgeActor.__name__:
			edge_decision, edge_probs = decision, probs
			
			# take out unrewarded_gradients, backward
			edge_gradient_dict = rl_trainer.on_action_taken(edge_decision, edge_probs, False)
			unrewarded_gradients.append(edge_gradient_dict)
			
			# decode trans_op
			corresponding_actor = rl_trainer.controller_net.actor(trans_op_type)
			if random:
				edge_decision = corresponding_actor.random_decision()
			
			edge_type = corresponding_actor.edge_candidates[edge_decision]
			# apply trans op
			parent_node = focus_node.parent
			edge_str = edge_type.split('_')[-1]
			if edge_str == 'Identity':
				new_edge_config = {
					'name': IdentityLayer.__name__,
					'ops_order': net_config.ops_order,
				}
			elif edge_str == 'DepthConv':
				depth_kernel_size = int(edge_type.split('x')[0])
				new_edge_config = {
					'name': DepthConvLayer.__name__,
					'kernel_size': depth_kernel_size,
					'ops_order': net_config.ops_order,
				}
				# if net_config.ops_order == 'bn_act_weight':
				# 	new_edge_config['bias'] = True
			elif edge_str == 'Conv':
				kernel_size = int(edge_type.split('x')[0])
				new_edge_config = {
					'name': ConvLayer.__name__,
					'kernel_size': kernel_size,
					'ops_order': net_config.ops_order,
				}
				# if net_config.ops_order == 'bn_act_weight':
				# 	new_edge_config['bias'] = True
			elif 'Pool' in edge_type:
				kernel_size = int(edge_type.split('x')[0])
				if edge_str == 'AVGPool':
					pool_type = 'avg'
				else:
					pool_type = 'max'
				new_edge_config = {
					'name': PoolingLayer.__name__,
					'pool_type': pool_type,
					'kernel_size': kernel_size,
					'stride': 1,
					'ops_order': net_config.ops_order,
				}
			elif 'FC' in edge_type:
				raise NotImplementedError
			else:
				raise ValueError
			op_param = {
				'idx': focus_node.idx,
				'edge_config': new_edge_config,
				'edge_type': edge_type,
			}
			trans_op = [TreeNode.REPLACE_IDENTITY_EDGE, parent_node.get_path_from_root, op_param]
			trans_ops_record.append(trans_op)
			
			parent_node.apply_trans_op(TreeNode.REPLACE_IDENTITY_EDGE, op_param)
		else:
			raise NotImplementedError
	
	# sum
	for _i in range(1, len(unrewarded_gradients)):
		target_unrewarded_grad, target_entropy_grad = unrewarded_gradients[0]
		src_unrewarded_grad, src_entropy_grad = unrewarded_gradients[_i]
		
		if target_entropy_grad is not None and src_entropy_grad is not None:
			RLTrainer.add_gradient_dict(target_entropy_grad, src_entropy_grad)
		
		if isinstance(target_unrewarded_grad, dict):
			RLTrainer.add_gradient_dict(target_unrewarded_grad, src_unrewarded_grad)
		else:
			for target, src in zip(target_unrewarded_grad, src_unrewarded_grad):
				RLTrainer.add_gradient_dict(target, src)
	unrewarded_gradients = unrewarded_gradients[0:1]
	
	return trans_ops_record, unrewarded_gradients, whole_tree


def arch_search_tree_cell(arch_search_folder, exp_settings, max_nets2train,
                          random=False, virtual=False, force_random_init=False):
	# reward config
	reward_config = OrderedDict([
		('func', 'tan'),  # 'tan'
		('merge', None),
		('decay', 0.95),
		('div_std', False),
	])
	
	# prepare arch_manager
	batch_num_per_update = 10
	arch_manager = ArchManager(arch_search_folder, exp_settings, batch_num_per_update, reward_config, random=random)
	
	#######################################################################################################
	
	# encoder net
	edge_candidates = [
		'3x3_DepthConv', '5x5_DepthConv', '7x7_DepthConv',
		'1x1_Conv',
		'Identity',
		'3x3_MAXPool', '3x3_AVGPool',
	]
	node_types = [
		'add-copy', 'concat-split', 'concat-copy'
	]
	tree_encoder_config = OrderedDict([
		('edge_vocab', Vocabulary(edge_candidates + ['3x3_GroupConv', '3x3_Conv'])),
		('node_vocab', Vocabulary(node_types)),
		('max_n', 2),
		('embedding_dim', 10),
		('hidden_size', 100),
		('bidirectional', True),
	])
	encoder_hidden_size = tree_encoder_config['hidden_size']
	if tree_encoder_config['bidirectional']: encoder_hidden_size *= 2
	
	if force_random_init:
		edge_candidates.append('3x3_Conv')
	# actor nets
	default_actor_config = [
		{'units': int(encoder_hidden_size * 2.5), 'use_bias': True, 'activation': 'relu'}
	]
	actor_dict = OrderedDict([
		(SetMergeTypeActor.__name__, {
			'merge_type_candidates': ['none', 'add', 'concat'],
			'branch_num_candidates': [
				[],
				[2, 3],
				[2],
			],
			'encoder_hidden_size': encoder_hidden_size,
			'actor_config': copy.deepcopy(default_actor_config),
		}),
		(InsertNodeActor.__name__, {
			'encoder_hidden_size': encoder_hidden_size,
			'actor_config': copy.deepcopy(default_actor_config),
		}),
		(ReplaceIdentityEdgeActor.__name__, {
			'edge_candidates': copy.deepcopy(edge_candidates),
			'encoder_hidden_size': encoder_hidden_size,
			'actor_config': copy.deepcopy(default_actor_config),
		}),
	])
	
	# build the rl_controller
	rl_meta_controller = RLTreeController(arch_manager.meta_controller_path, tree_encoder_config, actor_dict)
	
	#######################################################################################################
	
	encoder_init_scheme = {
		'embedding': {
			'type': 'uniform',
			'min': -0.1,
			'max': 0.1,
		},
		'cell': {
			'type': 'orthogonal',
		}
	}
	
	actor_init_scheme = {
		'type': 'default',
	}
	
	rl_meta_controller.tree_encoder.init_embedding(scheme=encoder_init_scheme['embedding'])
	rl_meta_controller.tree_encoder.init_cell(scheme=encoder_init_scheme['cell'])
	
	rl_meta_controller.init_actor(scheme=actor_init_scheme)
	
	rl_meta_controller.load()
	
	#######################################################################################################
	
	# rl_trainer config
	rl_trainer_config = {
		'algo': ReinforceTrainer.__name__,  # ReinforceTrainer
		'entropy_penalty': 1e-2,
		'opt_config': {
			'learning_rate': 6e-3,
			'opt_name': 'adam',
			'opt_param': {}
		}
	}
	
	# build rl trainer
	rl_trainer = get_rl_trainer_by_name(rl_trainer_config.pop('algo'))
	rl_trainer = rl_trainer(controller_net=rl_meta_controller, **rl_trainer_config)
	
	#######################################################################################################
	
	# network transformation config
	trans_noise_config = {
		'wider': {
			'type': 'normal',
			'ratio': 6e-3,
		},
		'deeper': {
			'type': 'normal',
			'ratio': 6e-3,
		},
		'bn': None
	}
	
	# run config during the validation phase
	run_config_for_validation = {
		'n_epochs': 20,  # 20
		'batch_size': 64,
		'init_lr': 0.035,
		'valid_size': 5000,
		'lr_schedule_type': 'cosine',
	}
	
	#######################################################################################################
	
	gpu_hours = 0
	
	arch_manager.trainer = rl_trainer
	# start architecture search
	while arch_manager.trained_nets < max_nets2train:
		# clear running queue
		while not len(arch_manager.running_queue) == 0:
			net_str, setting_idx, val = arch_manager.running_queue.pop()
			if isinstance(val, str):
				# not finished, back to <task_queue>
				arch_manager.task_queue.appendleft([net_str, setting_idx, val])
			elif isinstance(val, tuple):
				# finished, trigger <on_net_val_received>
				net_val, running_time = val
				arch_manager.on_net_val_received(net_str, setting_idx, net_val)
				gpu_hours += running_time / 60
		
		# sample cells and add to <task_queue>
		while len(arch_manager.task_queue) == 0:
			if len(arch_manager.result_slots) + len(arch_manager.update_queue) >= arch_manager.batch_size:
				if not random: break  # added for synchronized running
				
			net_config_for_sample, _, _ = arch_manager.get_start_net(setting_idx=0, _copy=False)
			
			# sample a new cell via transformation operations
			while True:
				trans_ops_record, unrewarded_gradients, cell_architecture = \
					sample_cell(net_config_for_sample.building_block, rl_trainer, net_config_for_sample,
					            trans_noise_config, random=random)
				if cell_architecture.height > 1:
					break
			
			# identifier of the sampled cell
			cell_str = cell_architecture.tree_str
			# check the sampled cell
			if cell_str in arch_manager.unrewarded_buffer:
				# on_running cell
				arch_manager.unrewarded_buffer[cell_str].append(unrewarded_gradients)
			else:
				assert arch_manager.result_slots.get(cell_str) is None
				arch_manager.unrewarded_buffer[cell_str] = [unrewarded_gradients]  # add to <unrewarded_buffer>
				
				to_run_setting = arch_manager.sample_exp_settings()  # get validation settings for the sampled cell
				slot_dict = arch_manager.result_slots[cell_str] = {}  # prepare results_slot
				for setting_idx in to_run_setting:
					net_val, net_folder = arch_manager.get_net_val(cell_str, setting_idx)
					if net_val is None:
						slot_dict[setting_idx] = net_folder
					else:
						slot_dict[setting_idx] = net_val
				
				if np.all([isinstance(val, float) for val in slot_dict.values()]):  # check if slots filled
					arch_manager.on_result_slots_filled(cell_str, is_new=False)
				else:
					for setting_idx, val in slot_dict.items():
						if not isinstance(val, str): continue
						# perform transformation operations
						net_config, run_config, data_provider = arch_manager.get_start_net(setting_idx, _copy=True)
						run_config.update(run_config_for_validation)  # change run_config for validation
						for block in net_config.blocks:
							if isinstance(block, DenseBlock) or isinstance(block, ResidualBlock) or \
									isinstance(block, ChainBlock):
								for op in trans_ops_record:
									op_type, node_path, op_param = copy.deepcopy(op)
									block.cell.apply_transformation(node_path, op_type, op_param)
						
						if not virtual:
							_, valid_loader, _ = data_provider
							valid_loader.batch_size = 200
							if torch.cuda.is_available():
								net_config = net_config.cuda()
							net_config.set_non_ready_layers(valid_loader, nBatch=10, noise=trans_noise_config,
							                                print_info=False)
						
						assert cell_architecture == Tree.build_tree_from_torch_module(net_config.building_block), 'Err'
						
						# prepare folder for validation
						if 'Pool' in cell_str or net_config.ops_order == 'bn_act_weight':
							mimic_run_config = {
								'src_model_dir': exp_settings[setting_idx][0],
							}
						else:
							mimic_run_config = None
						ExpdirMonitor.prepare_folder_for_valid(cell_str, net_config, run_config, val, trans_ops_record,
						                                       mimic_run_config=mimic_run_config,
						                                       no_init=force_random_init)
						# add to task_queue
						arch_manager.task_queue.appendleft([cell_str, setting_idx, val])
		
		if len(arch_manager.task_queue) > 0:
			if virtual:
				net_str, setting_idx, expdir = arch_manager.task_queue.pop()
				net_val = virtual_run(expdir)
				running_time = 0
				arch_manager.running_queue.appendleft([net_str, setting_idx, (net_val, running_time)])
			else:
				# get an idle gpu
				idle_gpu = arch_manager.gpu_cluster.get_idle_server()
				if idle_gpu is not None:
					net_str, setting_idx, expdir = arch_manager.task_queue.pop()
					idle_gpu.execute(net_str, setting_idx, expdir, arch_manager.running_queue)
				else:
					sleep(0.1)
		else:
			sleep(0.1)
		
	print('used gpu-hours: %s' % gpu_hours)


