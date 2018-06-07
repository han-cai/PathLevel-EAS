import os
import json
import subprocess
import argparse
from collections import OrderedDict
import operator

from expdir_monitor.expdir_monitor import hash_str2int


class NetPool:
	def __init__(self, path):
		self.path = os.path.realpath(path)
		os.makedirs(self.path, exist_ok=True)
		
		self.net_str2id = {}  # map net str to its id
		self.net_id2val = {}  # map id to corresponding value
		self.running_set = {}
		
		# load existing data
		if os.path.isfile(self.str2id_path):
			self.net_str2id = json.load(open(self.str2id_path, 'r'))
		if os.path.isfile(self.id2val_path):
			net_id2val = json.load(open(self.id2val_path, 'r'))
			for key in net_id2val:
				self.net_id2val[int(key)] = net_id2val[key]
		
		to_rename = []
		for folder in os.listdir(self.path):
			if folder.startswith('#'):  # indicate an expdir
				out_file = '%s/%s/output' % (self.path, folder)
				net_str_file = '%s/%s/net.str' % (self.path, folder)
				if not os.path.isfile(out_file) or not os.path.isfile(net_str_file):
					# remove the folder if it's an incomplete folder
					subprocess.run(['rm', '-rf', os.path.join(self.path, folder)])
				else:
					net_str = json.load(open(net_str_file, 'r'))['net_str']
					if self.net_str2id.get(net_str) is None:
						record = json.load(open(out_file, 'r'))
						if 'valid_acc' in record:
							net_val = float(record['valid_acc'])
						elif 'test_acc' in record:
							net_val = float(record['test_acc'])
						else:
							raise ValueError(record)
						net_id = self.add_net(net_str, net_val)  # add to the net pool
						folder_path = self.get_net_folder(net_id)  # set the folder_path according to net_id
					else:
						net_id = self.net_str2id[net_str]
						folder_path = self.get_net_folder(net_id)
					if folder_path != folder:
						to_rename.append([folder, folder_path])
		if len(to_rename) > 0:
			self.save()
		for src_folder, dst_folder in to_rename:
			try:
				os.rename(
					src=os.path.join(self.path, src_folder),
					dst=os.path.join(self.path, dst_folder),
				)
			except Exception:
				subprocess.run(['rm', '-rf', src_folder])
				
	@property
	def str2id_path(self):
		return '%s/str2id.pool' % self.path
	
	@property
	def id2val_path(self):
		return '%s/id2val.pool' % self.path
	
	@staticmethod
	def get_net_folder(net_id):
		# the folder name of a net with <net_id>
		return '#%s' % net_id
	
	def add_net(self, net_str, net_val, save=False):
		assert self.net_str2id.get(net_str) is None, '%s exists' % net_str
		net_id = hash_str2int(net_str)
		while net_id in self.net_id2val:
			net_id += 1
		self.net_str2id[net_str] = net_id
		self.net_id2val[net_id] = net_val
		if save:
			self.save()
		return net_id
	
	def get_net_val(self, net_str):
		net_id = self.net_str2id.get(net_str)
		if net_id is None:
			if net_str in self.running_set:
				running_id = self.running_set[net_str]
			else:
				running_id = hash_str2int(net_str)
				while running_id in self.running_set.values():
					running_id += 1
				self.running_set[net_str] = running_id
			net_folder = '%s/#Running_%s' % (self.path, running_id)
			return None, net_folder
		else:
			net_val = self.net_id2val[net_id]
			net_folder = '%s/%s' % (self.path, self.get_net_folder(net_id))
			return net_val, net_folder
	
	def save(self):
		sorted_id2val_items = sorted(self.net_id2val.items(), key=operator.itemgetter(1))
		sorted_id2val = OrderedDict(reversed(sorted_id2val_items))
		
		json.dump(self.net_str2id, open(self.str2id_path, 'w'), indent=4)
		json.dump(sorted_id2val, open(self.id2val_path, 'w'), indent=4)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str)
	
	args = parser.parse_args()
	if args.path is not None:
		net_pool = NetPool(args.path)
