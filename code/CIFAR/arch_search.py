import argparse
import numpy as np

import torch

from arch_search.arch_search_tree_cell import arch_search_tree_cell


parser = argparse.ArgumentParser()
parser.add_argument(
	'--setting', type=str, choices=['densenet-cell', 'pyramidnet-cell', 'chain-cell'],
)
parser.add_argument('--manual_seed', default=100, type=int)

exp_path = 'placeholder'
args = parser.parse_args()

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)

if args.setting == 'densenet-cell':
	folder_name = 'placeholder'
	arch_search_tree_cell(
		arch_search_folder=exp_path + 'Arch_Search/DenseNet/' + folder_name,
		exp_settings=[
			(exp_path + 'StartNets/DenseNet/' + folder_name,
			 exp_path + 'NetPool/DenseNet/' + folder_name)
		],
		max_nets2train=500,
		random=False,
		virtual=False,
	)
elif args.setting == 'pyramidnet-cell':
	folder_name = 'placeholder'
	arch_search_tree_cell(
		arch_search_folder=exp_path + 'Arch_Search/PyramidNet/' + folder_name,
		exp_settings=[
			(exp_path + 'StartNets/PyramidNet/' + folder_name,
			 exp_path + 'NetPool/PyramidNet/' + folder_name)
		],
		max_nets2train=500,
		random=False,
		virtual=False,
	)
elif args.setting == 'chain-cell':
	folder_name = 'placeholder'
	arch_search_tree_cell(
		arch_search_folder=exp_path + 'Arch_Search/ChainNet/' + folder_name,
		exp_settings=[
			(exp_path + 'StartNets/ChainNet/' + folder_name,
			 exp_path + 'NetPool/ChainNet/' + folder_name)
		],
		max_nets2train=300,
		random=False,
		virtual=False,
		force_random_init=True,
	)
else:
	print('Unsupported setting: %s' % args.setting)
