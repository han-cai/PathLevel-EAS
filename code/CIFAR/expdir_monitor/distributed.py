from subprocess import Popen, PIPE
from threading import Thread, Lock
from time import sleep
from sys import stderr
import re
import shlex
import numpy as np
from collections import deque


max_running_machine = 5
_max_used_mem = 0.5
_max_used_gpu = 0.5
config_file = 'server_config'


class GpuChecker:
	def __init__(self, nvidia_getter, gpuid):
		self.nvidia_getter = nvidia_getter
		self.gpuid = gpuid
	
	def _state_parser(self, state_str):
		result = []
		for line in state_str.split('\n'):
			pattern = re.search('.*?(\d*)C.*\|(.*?)MiB.*?/(.*?)MiB.*?\|.*?(\d*)%', line)
			if pattern is not None:
				result.append([int(x) for x in pattern.groups()])
		if self.gpuid >= len(result):
			return None
		# assert self.gpuid < len(result), 'Parsing error or not enough gpus.'
		return result[self.gpuid]
	
	def _instance_available(self, state_str):
		parse_result = self._state_parser(state_str)
		if parse_result is None: return False
		_, used_mem, total_mem, occupation = parse_result
		occupation /= 100
		return used_mem / total_mem < _max_used_mem and occupation < _max_used_gpu
	
	def check(self):
		_check_times = 3
		try:
			for _i in range(_check_times):
				assert self._instance_available(self.nvidia_getter())
				if _i < _check_times - 1:
					sleep(0.5)
		except AssertionError:
			return False
		return True


class RemoteController:
	def __init__(self, remote, gpuid, executive):
		self.remote = remote
		self.gpuid = gpuid
		self.executive = executive
		
		self.gpu_checker = GpuChecker(lambda: self.run('nvidia-smi', track=False), self.gpuid)
		self.alive_checker = lambda: self.run('echo "alive"', track=False)
		self.thread, self.process = None, None
		
		self._lock = Lock()
		self._occupied = False
	
	@property
	def occupied(self):
		with self._lock:
			return self._occupied
	
	@occupied.setter
	def occupied(self, val):
		assert isinstance(val, bool), 'Occupied must be True or False, but {} received.'.format(val)
		with self._lock:
			self._occupied = val
	
	def run(self, cmd, stdin=None, track=True):
		proc = Popen('ssh {} {}'.format(self.remote, shlex.quote(cmd)), shell=True, stdin=PIPE, stdout=PIPE,
		             universal_newlines=True)
		if track: self.process = proc
		return proc.communicate(input=stdin)[0]
	
	@property
	def gpu_state(self):
		return self.gpu_checker.check()
	
	@property
	def exe_cmd(self):
		return 'CUDA_VISIBLE_DEVICES={gpuid} python3 {executive}'.format(
			executive=self.executive,
			gpuid=self.gpuid
		)
	
	def check_on(self):
		try:
			feedback = self.alive_checker()
			assert 'alive' in feedback
			return True
		except AssertionError:
			if self.occupied:
				if self.thread.is_alive():
					self.process.terminate()
					self.thread.join()
				print('Remote {} is closed.'.format(self.remote), file=stderr)
			return False
	
	def _remote_executer(self, net_str, setting_idx, expdir, queue: deque):
		self.occupied = True
		
		cmd = self.exe_cmd
		print('{}: {} {}'.format(self.remote, cmd, expdir), file=stderr)
		result = self.run(cmd, stdin=expdir)
		try:
			result = str(result).split('\n')
			used_time = result[-3]
			result = result[-2]
			assert result.startswith('valid performance: ') and used_time.startswith('running time: '), \
				'Invalid return: %s, %s' % (used_time, result)
			used_time = used_time[len('running time: '):]
			used_time = float(used_time) / 60  # minutes
			
			result = result[len('valid performance: '):]
			result = float(result)
			queue.appendleft([net_str, setting_idx, (result, used_time)])
			print('task {} is successfully executed, result is {}, using {} min.'.
			      format(expdir, result, used_time), file=stderr)
		except Exception:
			queue.appendleft([net_str, setting_idx, expdir])
			print('task {} fails, with return: %s.'.format(expdir, result), file=stderr)
		self.occupied = False
	
	def execute(self, net_str, setting_idx, expdir, queue: deque):
		self.thread = Thread(target=self._remote_executer, args=(net_str, setting_idx, expdir, queue))
		self.thread.start()


class ClusterController:
	def __init__(self, config_list):
		self.cluster = [RemoteController(*config) for config in config_list]
		self._pt = 0
	
	@property
	def remote_num(self):
		return len(self.cluster)
	
	def _get_available_servers(self):
		occupy_num = 0
		available_servers = [False] * self.remote_num
		for _i, remote in enumerate(self.cluster):
			if remote.check_on():
				if remote.occupied:
					occupy_num += 1
				else:
					available_servers[_i] = True
		return available_servers, occupy_num
	
	def get_idle_server(self):
		""" not block """
		available_servers, occupy_num = self._get_available_servers()
		choose_remote = None
		if occupy_num < max_running_machine and np.any(available_servers):
			for _i in range(0, self.remote_num):
				try_pt, self._pt = self._pt, (self._pt + 1) % self.remote_num
				if available_servers[try_pt] and self.cluster[try_pt].gpu_state:
					choose_remote = self.cluster[try_pt]
					break
		return choose_remote
