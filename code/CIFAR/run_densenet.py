import argparse
import os
import json
import numpy as np

import torch

from models.run_manager import RunConfig, RunManger
from models.densenet import DenseNet


run_config_cifar = {
	'n_epochs': 300,
	'batch_size': 64,
	'opt_type': 'sgd',
	'opt_param': {'momentum': 0.9, 'nesterov': True},
	'weight_decay': 1e-4,
	######################################################
	'init_lr': 0.1,
	'lr_schedule_type': 'cosine',
	'lr_schedule_param': {'reduce_epochs': [0.5, 0.75], 'reduce_factors': [10, 10]},
	######################################################
	'model_init': 'he_fout',  # he_fin, he_fout
	'init_div_groups': True,
	'validation_frequency': 10,
	'renew_logs': False,
	######################################################
	'dataset': None,
	'use_torch_data_loader': False,
	######################################################
	'valid_size': None,
	'norm_before_aug': True,
	'flip_first': False,
	'drop_last': True,
	'include_extra': None,
	######################################################
	'cutout': False,
	'cutout_n_holes': 1,
	'cutout_size': 16,
}

model_config = {
	'growth_rate': 48,
	'dense_block_per_group': 1,
	'total_groups': 3,
	'downsample_type': 'conv-pool',  # conv-pool, pool
	######################################################
	'first_ratio': 2,
	'reduction': 0.5,
	'bottleneck': 4,
	'final_bn': False,
	'no_first_relu': False,
	'use_depth_sep_conv': False,
	'groups_3x3': 2,
	######################################################
	'ops_order': 'bn_act_weight',  # bn_act_weight, weight_bn_act, act_weight_bn
	'dropout_rate': 0,
	######################################################
	'path_drop_rate': 0,
	'use_zero_drop': True,
	'drop_only_add': False
}


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default='', help='path to dataset')
	parser.add_argument('--manual_seed', default=0, type=int)
	parser.add_argument(
		'--dataset', type=str, default='C10+', choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
	)
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--gpu', help='gpu available', default='0')
	
	args = parser.parse_args()
	
	torch.manual_seed(args.manual_seed)
	torch.cuda.manual_seed_all(args.manual_seed)
	np.random.seed(args.manual_seed)
	
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	if args.dataset in ['C10', 'C100', 'C10+', 'C100+']:
		run_config_cifar['dataset'] = args.dataset
		run_config = RunConfig(**run_config_cifar)
		data_shape = (3, 32, 32)
		n_classes = 10
	else:
		raise NotImplementedError
	
	if len(args.path) == 0:
		args.path = '../../Exp/TrainedNets/DenseNet' \
		            '/%s/L=%d_K=%d' % \
		            (run_config.dataset, model_config['dense_block_per_group'], model_config['growth_rate'])
		args.path += '#valid=%s' % run_config.valid_size
		if model_config['use_depth_sep_conv']:
			args.path += '#depth_conv'
		else:
			args.path += '#groups3x3=%d' % model_config['groups_3x3']
		args.path += '#' + os.uname()[1]
		
	# print configurations
	print('Run config:')
	for k, v in run_config.get_config().items():
		print('\t%s: %s' % (k, v))
	print('Network config:')
	for k, v in model_config.items():
		print('\t%s: %s' % (k, v))

	model = DenseNet.set_standard_net(data_shape=data_shape, n_classes=n_classes, **model_config)
	run_manager = RunManger(args.path, model, run_config, out_log=True, resume=True)
	
	run_manager.save_config()
	if args.train:
		run_manager.train()
	
	loss, acc = run_manager.validate()
	test_log = 'test_loss: %f\t test_acc: %f' % (loss, acc)
	run_manager.write_log(test_log, prefix='test')
	json.dump({'test_loss': '%f' % loss, 'test_acc': '%f' % acc}, open('%s/output' % args.path, 'w'))
	
	run_manager.save_model()

