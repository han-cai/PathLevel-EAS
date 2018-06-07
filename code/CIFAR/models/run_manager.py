import os
import time
import shutil
import json
from datetime import timedelta

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import DataProvider
from data_providers.cifar import Cifar10DataProvider
from data_providers.utils import get_data_provider_by_name
from models.densenet import DenseNet
from models.pyramidnet import PyramidNet
from models.chain_net import ChainNet
from models.utils import *


def get_model_by_name(name):
	if name == DenseNet.__name__:
		return DenseNet
	elif name == PyramidNet.__name__:
		return PyramidNet
	elif name == ChainNet.__name__:
		return ChainNet
	else:
		raise ValueError('unrecognized model name: %s' % name)


class RunConfig:
	def __init__(self, n_epochs=300, batch_size=64, opt_type='sgd', opt_param=None, weight_decay=1e-4,
	             init_lr=0.1, lr_schedule_type='cosine', lr_schedule_param=None,
	             model_init='he_fout', init_div_groups=True,
	             validation_frequency=10, renew_logs=False,
	             dataset='C10+', use_torch_data_loader=False,
	             valid_size=None, norm_before_aug=True, flip_first=False, drop_last=True, include_extra=False,
	             cutout=False, cutout_n_holes=1, cutout_size=16):
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.opt_type = opt_type
		self.opt_param = opt_param
		self.weight_decay = weight_decay
		
		self.init_lr = init_lr
		self.lr_schedule_type = lr_schedule_type
		self.lr_schedule_param = lr_schedule_param
		
		self.model_init = model_init
		self.init_div_groups = init_div_groups
		
		self.validation_frequency = validation_frequency
		self.renew_logs = renew_logs
		
		self.dataset = dataset
		self.use_torch_data_loader = use_torch_data_loader
		self.valid_size = valid_size
		self.norm_before_aug = norm_before_aug
		self.flip_first = flip_first
		self.drop_last = drop_last
		self.include_extra = include_extra
		
		self.cutout = cutout
		self.cutout_n_holes = cutout_n_holes
		self.cutout_size = cutout_size
	
	def get_config(self):
		return self.__dict__.copy()
	
	def update(self, attributes):
		self.__dict__.update(attributes)
	
	def copy(self):
		return RunConfig(**self.get_config())
	
	@staticmethod
	def get_default_run_config(dataset='C10+'):
		default_run_config = RunConfig()
		default_run_config.opt_param = {'momentum': 0.9, 'nesterov': True}
		default_run_config.dataset = dataset
		return default_run_config.get_config()
	
	def _learning_rate(self, epoch, batch=0, nBatch=None):
		if self.lr_schedule_type == 'cosine':
			T_total = self.n_epochs * nBatch
			T_cur = epoch * nBatch + batch
			lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
		else:
			param = {} if self.lr_schedule_param is None else self.lr_schedule_param
			reduce_epochs = param.get('reduce_epochs', [0.5, 0.75])
			reduce_factors = param.get('reduce_factors', [10, 10])
			lr = self.init_lr
			for _reduce_epoch, _reduce_factor in zip(reduce_epochs, reduce_factors):
				if epoch >= _reduce_epoch * self.n_epochs:
					lr /= _reduce_factor
		return lr
	
	def build_data_provider(self):
		if self.use_torch_data_loader:
			if self.dataset == 'C10+':
				mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
				stdv = [x / 255.0 for x in [63.0, 62.1, 66.7]]
				
				train_transforms = transforms.Compose([
					transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(mean=mean, std=stdv),
				])
				if self.cutout:
					train_transforms.transforms.append(Cutout(n_holes=self.cutout_n_holes, length=self.cutout_size))
					
				test_transforms = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize(mean=mean, std=stdv),
				])
				data_path = Cifar10DataProvider.default_data_path()
				train_set = datasets.CIFAR10(data_path, train=True, transform=train_transforms, download=True)
				test_set = datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=False)
				if self.valid_size is not None:
					valid_set = datasets.CIFAR10(data_path, train=True, transform=test_transforms, download=True)
					np.random.seed(DataProvider.SEED)  # set random seed before sampling validation set
					
					indices = np.random.permutation(len(train_set))
					train_indices = indices[self.valid_size:]
					train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
					valid_indices = indices[:self.valid_size]
					valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
					
					train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
					                                           sampler=train_sampler,
					                                           pin_memory=cuda_available(), num_workers=1,
					                                           drop_last=self.drop_last)
					valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size,
					                                           sampler=valid_sampler,
					                                           pin_memory=cuda_available(), num_workers=1,
					                                           drop_last=False)
				else:
					train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
					                                           pin_memory=cuda_available(), num_workers=1,
					                                           drop_last=self.drop_last)
					valid_loader = None
				
				test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False,
				                                          pin_memory=cuda_available(), num_workers=1,
				                                          drop_last=False)
				if valid_loader is None:
					valid_loader = test_loader
			else:
				raise NotImplementedError
		else:
			data_provider = get_data_provider_by_name(self.dataset, self.get_config())
			train_loader = data_provider.train
			valid_loader = data_provider.validation
			test_loader = data_provider.test
			
			train_loader.batch_size = self.batch_size
			valid_loader.batch_size = self.batch_size
			test_loader.batch_size = 100
		return train_loader, valid_loader, test_loader
	
	def build_optimizer(self, params):
		if self.opt_type == 'sgd':
			opt_param = {} if self.opt_param is None else self.opt_param
			momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
			optimizer = torch.optim.SGD(params, self.init_lr, momentum=momentum, nesterov=nesterov,
			                            weight_decay=self.weight_decay)
		else:
			raise NotImplementedError
		return optimizer
	
	def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
		"""adjust learning of a given optimizer and return the new learning rate"""
		new_lr = self._learning_rate(epoch, batch, nBatch)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		return new_lr


class RunManger:
	def __init__(self, path, model, run_config: RunConfig, out_log=True, resume=True):
		self.path = path
		self.model = model
		self.run_config = run_config
		self.out_log = out_log
		self.resume = resume
		
		self._logs_path, self._save_path = None, None
		self._best_acc = 0
		self._start_epoch = 0
		
		total_params = self.count_parameters(self.model)
		if self.out_log:
			print('Total training params: %.2fM' % (total_params / 1e6))
		with open('%s/net_info.txt' % self.logs_path, 'a') as fout:
			fout.write(json.dumps({'param': '%.2fM' % (total_params / 1e6)}) + '\n')
		
		# prepare data loader
		self.train_loader, self.valid_loader, self.test_loader = self.run_config.build_data_provider()
		
		# initialize model (default)
		self.model.init_model(run_config.model_init, run_config.init_div_groups)
		
		# prepare model (move to GPU if available)
		if cuda_available():
			if torch.cuda.device_count() > 1:
				self.model = torch.nn.DataParallel(self.model).cuda()
			else:
				self.model = self.model.cuda()
			cudnn.benchmark = True
		if self.out_log:
			print(self.model)
		
		# prepare optimizer and criterion
		if cuda_available():
			self.criterion = nn.CrossEntropyLoss().cuda()
		else:
			self.criterion = nn.CrossEntropyLoss()
		self.optimizer = self.run_config.build_optimizer(self.model.parameters())
		
		# load model
		if self.resume:
			self.load_model()
	
	@staticmethod
	def count_parameters(model):
		total_params = sum([p.data.nelement() for p in model.parameters()])
		return total_params
		
	@property
	def save_path(self):
		if self._save_path is None:
			save_path = '%s/checkpoint' % self.path
			os.makedirs(save_path, exist_ok=True)
			self._save_path = save_path
		return self._save_path
	
	@property
	def logs_path(self):
		if self._logs_path is None:
			logs_path = '%s/logs' % self.path
			if self.run_config.renew_logs:
				shutil.rmtree(logs_path, ignore_errors=True)
			os.makedirs(logs_path, exist_ok=True)
			self._logs_path = logs_path
		return self._logs_path
	
	def save_model(self, checkpoint=None, is_best=False):
		if checkpoint is None:
			checkpoint = {'state_dict': self.model.state_dict()}
			
		checkpoint['dataset'] = self.run_config.dataset
		latest_fname = os.path.join(self.save_path, 'latest.txt')
		model_path = os.path.join(self.save_path, 'checkpoint.pth.tar')
		with open(latest_fname, 'w') as fout:
			fout.write(model_path + '\n')
		torch.save(checkpoint, model_path)
		
		if is_best:
			best_path = os.path.join(self.save_path, 'model_best.pth.tar')
			shutil.copyfile(model_path, best_path)
	
	def load_model(self, model_fname=None):
		latest_fname = os.path.join(self.save_path, 'latest.txt')
		if model_fname is None and os.path.exists(latest_fname):
			with open(latest_fname, 'r') as fin:
				model_fname = fin.readline()
				if model_fname[-1] == '\n':
					model_fname = model_fname[:-1]
		try:
			if not os.path.exists(model_fname):
				model_fname = '%s/checkpoint.pth.tar' % self.save_path
				with open(latest_fname, 'w') as fout:
					fout.write(model_fname + '\n')
			if self.out_log:
				print("=> loading checkpoint '{}'".format(model_fname))
			if cuda_available():
				checkpoint = torch.load(model_fname)
			else:
				checkpoint = torch.load(model_fname, map_location='cpu')
			if 'epoch' in checkpoint:
				self._start_epoch = checkpoint['epoch'] + 1
			if 'best_acc' in checkpoint:
				self._best_acc = checkpoint['best_acc']
			if 'optimizer' in checkpoint:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.model.load_state_dict(checkpoint['state_dict'])
			if self.out_log:
				print("=> loaded checkpoint '{}'".format(model_fname))
			# set new manual seed
			new_manual_seed = int(time.time())
			torch.manual_seed(new_manual_seed)
			torch.cuda.manual_seed_all(new_manual_seed)
			np.random.seed(new_manual_seed)
		except Exception:
			if self.out_log:
				print('fail to load checkpoint from %s' % self.save_path)
	
	def write_log(self, log_str, prefix, should_print=True):
		"""prefix: valid, train, test"""
		if prefix in ['valid', 'test']:
			with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
				fout.write(log_str + '\n')
				fout.flush()
		if prefix in ['valid', 'test', 'train']:
			with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
				if prefix in ['valid', 'test']:
					fout.write('=' * 10)
				fout.write(log_str + '\n')
				fout.flush()
		if should_print:
			print(log_str)
	
	def train(self):
		data_loader = self.train_loader
		
		for epoch in range(self._start_epoch, self.run_config.n_epochs):
			print('\n', '-' * 30, 'Train epoch: %d' % epoch, '-' * 30, '\n')
			
			batch_time = AverageMeter()
			losses = AverageMeter()
			top1 = AverageMeter()
			
			# switch to train mode
			self.model.train()
			
			end = time.time()
			for i, (_input, target) in enumerate(data_loader):
				lr = self.run_config.adjust_learning_rate(self.optimizer, epoch, batch=i, nBatch=len(data_loader))
				
				if cuda_available():
					target = target.cuda(async=True)
					_input = _input.cuda()
				input_var = torch.autograd.Variable(_input)
				target_var = torch.autograd.Variable(target)
				
				# compute output
				output = self.model(input_var)
				loss = self.criterion(output, target_var)
				
				# measure accuracy and record loss
				acc1 = accuracy(output.data, target, topk=(1,))
				losses.update(loss.data[0], _input.size(0))
				top1.update(acc1[0][0], _input.size(0))
				
				# compute gradient and do SGD step
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()
				
				if i % 70 == 0 or i + 1 == len(data_loader):
					batch_log = 'Train [{0}][{1}/{2}]\t' \
					            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
					            'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
					            'top 1-acc {top1.val:.3f} ({top1.avg:.3f})\tlr {lr:.5f}'.\
						format(epoch, i, len(data_loader), batch_time=batch_time, losses=losses, top1=top1, lr=lr)
					self.write_log(batch_log, 'train')
					
			time_per_epoch = batch_time.sum
			seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
			print('Time per epoch: %s, Est. complete in: %s' % (
				str(timedelta(seconds=time_per_epoch)),
				str(timedelta(seconds=seconds_left))))
			
			if (epoch + 1) % self.run_config.validation_frequency == 0:
				val_loss, val_acc = self.validate(use_test_loader=False)
				is_best = val_acc > self._best_acc
				self._best_acc = max(self._best_acc, val_acc)
				val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\tacc {3:.3f} ({4:.3f})\t'. \
					format(epoch, self.run_config.n_epochs, val_loss, val_acc, self._best_acc)
				self.write_log(val_log, 'valid')
			else:
				is_best = False
			self.save_model({
				'epoch': epoch,
				'best_acc': self._best_acc,
				'optimizer': self.optimizer.state_dict(),
				'state_dict': self.model.state_dict(),
			}, is_best)
	
	def validate(self, use_test_loader=True):
		if use_test_loader:
			data_loader = self.test_loader
		else:
			data_loader = self.valid_loader
		self.model.eval()
		
		losses = AverageMeter()
		top1 = AverageMeter()
		
		for i, (_input, target) in enumerate(data_loader):
			if cuda_available():
				target = target.cuda(async=True)
				_input = _input.cuda()
			input_var = torch.autograd.Variable(_input, volatile=True)
			target_var = torch.autograd.Variable(target, volatile=True)
	
			# compute output
			output = self.model(input_var)
			loss = self.criterion(output, target_var)
	
			# measure accuracy and record loss
			acc1 = accuracy(output.data, target, topk=(1,))
			losses.update(loss.data[0], _input.size(0))
			top1.update(acc1[0][0], _input.size(0))
		return losses.avg, top1.avg
	
	def pure_train(self):
		data_loader = self.train_loader
		
		for epoch in range(self._start_epoch, self.run_config.n_epochs):
			# switch to train mode
			self.model.train()
			for i, (_input, target) in enumerate(data_loader):
				_ = self.run_config.adjust_learning_rate(self.optimizer, epoch, batch=i, nBatch=len(data_loader))
				
				if cuda_available():
					target = target.cuda(async=True)
					_input = _input.cuda()
				input_var = torch.autograd.Variable(_input)
				target_var = torch.autograd.Variable(target)
				
				# compute output
				output = self.model(input_var)
				loss = self.criterion(output, target_var)
				
				# compute gradient and do SGD step
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
		
	def save_config(self, print_info=True):
		""" dump run_config and net_config to the model_folder """
		os.makedirs(self.path, exist_ok=True)
		net_save_path = os.path.join(self.path, 'net.config')
		if isinstance(self.model, nn.DataParallel):
			model_config = self.model.module.get_config()
		else:
			model_config = self.model.get_config()
		json.dump(model_config, open(net_save_path, 'w'), indent=4)
		if print_info: print('Network configs dump to %s' % self.save_path)
		
		run_save_path = os.path.join(self.path, 'run.config')
		json.dump(self.run_config.get_config(), open(run_save_path, 'w'), indent=4)
		if print_info: print('Run configs dump to %s' % run_save_path)

