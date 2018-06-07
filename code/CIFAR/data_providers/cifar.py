import os
import pickle
import random
import numpy as np
import math
import tempfile

from data_providers.base_provider import ImagesDataSet, DataProvider
from data_providers.downloader import download_data_url
from models.utils import Cutout


def random_horizontal_flip(image):
	flip = random.getrandbits(1)
	if flip:
		image = image[:, ::-1, :]
	return image


def random_crop(image, pad):
	init_shape = image.shape
	new_shape = [init_shape[0] + pad * 2,
	             init_shape[1] + pad * 2,
	             init_shape[2]]
	zeros_padded = np.zeros(new_shape)
	zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
	
	# randomly crop to original size
	init_x = np.random.randint(0, pad * 2)
	init_y = np.random.randint(0, pad * 2)
	cropped = zeros_padded[
	          init_x: init_x + init_shape[0],
	          init_y: init_y + init_shape[1],
	          :]
	return cropped


def augment_image(image, pad, flip_first):
	""" Perform zero padding, randomly crop image to original size, maybe mirror horizontally """
	if flip_first:
		image = random_horizontal_flip(image)
		image = random_crop(image, pad)
	else:
		image = random_crop(image, pad)
		image = random_horizontal_flip(image)
	return image


def augment_all_images(initial_images, pad=4, flip_first=True):
	new_images = np.zeros(initial_images.shape)
	for i in range(initial_images.shape[0]):
		new_images[i] = augment_image(initial_images[i], pad, flip_first)
	return new_images


class CifarDataSet(ImagesDataSet):
	def __init__(self, images, labels, shuffle, augmentation, meanstd, norm_before_aug,
	             flip_first=True, drop_last=True, cutout=None):
		super(CifarDataSet, self).__init__()
		
		self._batch_counter, self.epoch_images, self.epoch_labels = 0, None, None
		
		self.images = images
		self.labels = labels
		self.shuffle = shuffle
		self.augmentation = augmentation
		self.mean_std = meanstd
		self.norm_before_aug = norm_before_aug
		
		self.flip_first = flip_first
		self.drop_last = drop_last
		self.cutout = cutout
		
		if not self.augmentation or self.norm_before_aug:
			self.images = self.normalize_images(images, 'by_channels', self.mean_std)
		self.start_new_epoch()
	
	@property
	def num_examples(self):
		return self.labels.shape[0]
	
	def batch_num(self, batch_size):
		if self.drop_last:
			batch_num = self.num_examples // batch_size
		else:
			batch_num = int(math.ceil(self.num_examples / batch_size))
		return batch_num
	
	def start_new_epoch(self):
		self._batch_counter = 0
		if self.shuffle:
			images, labels = self.shuffle_images_and_labels(self.images, self.labels)
		else:
			images, labels = self.images, self.labels
		if self.augmentation:
			images = augment_all_images(images, pad=4, flip_first=self.flip_first)
			if not self.norm_before_aug:
				images = self.normalize_images(images, 'by_channels', self.mean_std)
		self.epoch_images = np.transpose(images, axes=[0, 3, 1, 2])  # N, C, H, W
		self.epoch_labels = labels
		if self.cutout:
			for _i in range(len(self.epoch_images)):
				self.epoch_images[_i] = self.cutout(self.epoch_images[_i])
	
	def next_batch(self, batch_size):
		start = self._batch_counter * batch_size
		end = (self._batch_counter + 1) * batch_size
		self._batch_counter += 1
		images_slice = self.epoch_images[start: end]
		labels_slice = self.epoch_labels[start: end]
		if images_slice.shape[0] != batch_size:
			if self.drop_last or images_slice.shape[0] == 0:
				self.start_new_epoch()
				return self.next_batch(batch_size)
		
		return images_slice, labels_slice
	

class CifarDataProvider(DataProvider):
	""" Abstract class for cifar readers """
	
	def __init__(self, save_path=None, one_hot=False,
	             valid_size=None, norm_before_aug=True, flip_first=True, drop_last=True,
	             cutout=False, cutout_n_holes=1, cutout_size=16, **kwargs):
		
		self._save_path = save_path
		
		download_data_url(self.data_url, self.save_path)
		train_fnames, test_fnames = self.get_filenames(self.save_path)
		
		# add train and validations datasets
		train_images, train_labels = self.read_cifar(train_fnames, one_hot)
		train_meanstd = ImagesDataSet.measure_mean_and_std(train_images)
		if valid_size is not None:
			np.random.seed(DataProvider.SEED)  # set random seed before sampling validation set
			train_images, train_labels = ImagesDataSet.shuffle_images_and_labels(train_images, train_labels)
			
			valid_images, valid_labels = train_images[:valid_size], train_labels[:valid_size]
			train_images, train_labels = train_images[valid_size:], train_labels[valid_size:]
			
			self.validation = CifarDataSet(
				images=valid_images, labels=valid_labels, shuffle=False, augmentation=False, meanstd=train_meanstd,
				norm_before_aug=norm_before_aug, flip_first=flip_first, drop_last=False, cutout=None
			)
		
		if cutout:
			train_cutout = Cutout(n_holes=cutout_n_holes, length=cutout_size)
		else:
			train_cutout = None
		self.train = CifarDataSet(
			images=train_images, labels=train_labels, shuffle=True, augmentation=self.data_augmentation,
			meanstd=train_meanstd, norm_before_aug=norm_before_aug, flip_first=flip_first, drop_last=drop_last,
			cutout=train_cutout
		)
		
		# add test set
		test_images, test_labels = self.read_cifar(test_fnames, one_hot)
		self.test = CifarDataSet(
			images=test_images, labels=test_labels, shuffle=False, augmentation=False, meanstd=train_meanstd,
			norm_before_aug=norm_before_aug, flip_first=flip_first, drop_last=False, cutout=None
		)
		
		if valid_size is None:
			self.validation = self.test
	
	def read_cifar(self, filenames, one_hot):
		if self.n_classes == 10:
			labels_key = b'labels'
		elif self.n_classes == 100:
			labels_key = b'fine_labels'
		else:
			raise ValueError
		
		images_res = []
		labels_res = []
		for fname in filenames:
			with open(fname, 'rb') as f:
				images_and_labels = pickle.load(f, encoding='bytes')
			images = images_and_labels[b'data']
			images = images.reshape(-1, 3, 32, 32)
			images = images.swapaxes(1, 3).swapaxes(1, 2)
			images_res.append(images)
			labels_res.append(images_and_labels[labels_key])
		images_res = np.vstack(images_res)
		labels_res = np.hstack(labels_res)
		if one_hot:
			labels_res = self.labels_to_one_hot(labels_res)
		return images_res, labels_res
	
	@property
	def data_shape(self):
		return 3, 32, 32  # C, H, W
	
	@property
	def n_classes(self):
		# return self._n_classes
		raise NotImplementedError
	
	@property
	def save_path(self):
		if self._save_path is None:
			self._save_path = self.default_data_path()
		return self._save_path
	
	@property
	def data_url(self):
		""" Return url for downloaded data depends on cifar class """
		data_url = ('http://www.cs.toronto.edu/'
		            '~kriz/cifar-%d-python.tar.gz' % self.n_classes)
		return data_url
	
	@property
	def data_augmentation(self):
		raise NotImplementedError
	
	def get_filenames(self, save_path):
		""" Return two lists of train and test filenames for dataset """
		raise NotImplementedError
	
	@staticmethod
	def default_data_path():
		raise NotImplementedError


class Cifar10DataProvider(CifarDataProvider):
	@property
	def n_classes(self):
		return 10
	
	@property
	def data_augmentation(self):
		return False
	
	def get_filenames(self, save_path):
		sub_save_path = os.path.join(save_path, 'cifar-10-batches-py')
		train_filenames = [
			os.path.join(
				sub_save_path,
				'data_batch_%d' % i) for i in range(1, 6)]
		test_filenames = [os.path.join(sub_save_path, 'test_batch')]
		return train_filenames, test_filenames
	
	@staticmethod
	def default_data_path():
		return os.path.join(tempfile.gettempdir(), 'datasets/cifar10')


class Cifar100DataProvider(CifarDataProvider):
	@property
	def n_classes(self):
		return 100
	
	@property
	def data_augmentation(self):
		return False
	
	def get_filenames(self, save_path):
		sub_save_path = os.path.join(save_path, 'cifar-100-python')
		train_filenames = [os.path.join(sub_save_path, 'train')]
		test_filenames = [os.path.join(sub_save_path, 'test')]
		return train_filenames, test_filenames
	
	@staticmethod
	def default_data_path():
		return os.path.join(tempfile.gettempdir(), 'datasets/cifar100')


class Cifar10AugmentedDataProvider(Cifar10DataProvider):
	@property
	def data_augmentation(self):
		return True


class Cifar100AugmentedDataProvider(Cifar100DataProvider):
	@property
	def data_augmentation(self):
		return True
