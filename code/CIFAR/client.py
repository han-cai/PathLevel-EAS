"""
The file to run in the client side
Train the network and return the validation performance
"""

import time
import numpy as np
import os
import json

import torch

from expdir_monitor.expdir_monitor import ExpdirMonitor


def run(expdir):
	manual_seed = 0
	torch.manual_seed(manual_seed)
	torch.cuda.manual_seed_all(manual_seed)
	np.random.seed(manual_seed)
	
	start_time = time.time()
	expdir_monitor = ExpdirMonitor(expdir)
	if os.path.isfile('%s/mimic_run' % expdir):
		mimic_run_param = json.load(open('%s/mimic_run' % expdir, 'r'))
		mimic_run_param.update({
			'sample_size': 1000,
			'distill_epochs': 30,
			'distill_lr': 0.02,
		})
		expdir_monitor.mimic_run(**mimic_run_param)
	valid_performance = expdir_monitor.run(pure=True, train=False, use_test_loader=False, resume=False)
	end_time = time.time()
	print('running time: %s' % (end_time - start_time))
	print('valid performance: %f' % valid_performance)


def main():
	expdir = input().strip('\n')
	run(expdir)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
