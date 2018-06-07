"""
Given a expdir, run the exp
"""

import argparse
import numpy as np
import os

import torch

from expdir_monitor.expdir_monitor import ExpdirMonitor


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str)
	parser.add_argument('--manual_seed', default=0, type=int)
	
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--valid', action='store_true')
	parser.add_argument('--valid_size', default=None, type=int)
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--gpu', help='gpu available', default='0')
	
	args = parser.parse_args()
	
	torch.manual_seed(args.manual_seed)
	torch.cuda.manual_seed_all(args.manual_seed)
	np.random.seed(args.manual_seed)
	
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	expdir_monitor = ExpdirMonitor(args.path)
	expdir_monitor.run(pure=False, train=args.train, use_test_loader=(not args.valid), valid_size=args.valid_size,
	                   resume=args.resume)


