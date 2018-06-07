import torch

import numpy as np


class DataSet:
	""" Class to represent some dataset: train, validation, test """
	
	def __init__(self):
		self._iter_batch_size = None
		self._iter_pt = 0
	
	@property
	def num_examples(self):
		""" Return number of examples in dataset """
		raise NotImplementedError
	
	def batch_num(self, batch_size):
		""" number of batch per epoch """
		raise NotImplementedError
	
	def start_new_epoch(self):
		""" reset batch_counter, epoch_images and epoch_labels """
		raise NotImplementedError
	
	def next_batch(self, batch_size):
		""" Return batch of required size of data, labels """
		raise NotImplementedError
	
	@property
	def batch_size(self):
		return self._iter_batch_size
	
	@batch_size.setter
	def batch_size(self, value):
		self._iter_batch_size = value
		
	def __iter__(self):
		return self
	
	def __len__(self):
		if self._iter_batch_size is None:
			raise ValueError('Must set iter_batch_size')
		return self.batch_num(self._iter_batch_size)
	
	def __next__(self):
		if self._iter_batch_size is None:
			raise ValueError('Must set iter_batch_size')
		if self._iter_pt >= len(self):
			self._iter_pt = 0
			raise StopIteration
		self._iter_pt += 1
		_input, target = self.next_batch(self._iter_batch_size)
		return _input, target


class ImagesDataSet(DataSet):
	""" Dataset for images that provide some often used methods """
	
	@property
	def num_examples(self):
		""" Return number of examples in dataset """
		raise NotImplementedError
	
	def batch_num(self, batch_size):
		""" number of batch per epoch """
		raise NotImplementedError
	
	def start_new_epoch(self):
		""" reset batch_counter, epoch_images and epoch_labels """
		raise NotImplementedError
	
	def next_batch(self, batch_size):
		""" Return batch of required size of data, labels """
		raise NotImplementedError
	
	def __next__(self):
		_input, target = super(ImagesDataSet, self).__next__()
		return torch.from_numpy(_input).float(), torch.from_numpy(target)
	
	@staticmethod
	def measure_mean_and_std(images):
		""" compute the channel means and stds of the input images: [, width, height, channel] """
		# for every channel in image
		means = []
		stds = []
		# for every channel in image (assume this is last dimension)
		for ch in range(images.shape[-1]):
			means.append(np.mean(images[:, :, :, ch]))
			stds.append(np.std(images[:, :, :, ch]))
		return means, stds
	
	@staticmethod
	def shuffle_images_and_labels(images, labels):
		""" random permutation: images and labels """
		rand_indexes = np.random.permutation(images.shape[0])
		shuffled_images = images[rand_indexes]
		shuffled_labels = labels[rand_indexes]
		return shuffled_images, shuffled_labels
	
	@staticmethod
	def normalize_images(images, normalization_type, meanstd=None):
		"""
		Args:
			images: numpy 4D array [batch, width, height, channel]
			normalization_type: `str`, available choices:
				- divide_255
				- divide_256
				- by_channels
			meanstd: [<mean>, <std>]
		"""
		if normalization_type is not None:
			if normalization_type == 'divide_255':
				images = images / 255
			elif normalization_type == 'divide_256':
				images = images / 256
			elif normalization_type == 'by_channels':
				images = images.astype(np.float32)
				if meanstd is None:
					# use the mean and std of the input images if meanstd is not given
					meanstd = ImagesDataSet.measure_mean_and_std(images)
				means, stds = meanstd
				# for every channel in image (assume this is last dimension)
				for ch in range(images.shape[-1]):
					images[:, :, :, ch] = (images[:, :, :, ch] - means[ch]) / stds[ch]
			else:
				raise Exception('Unknown type of normalization')
		return images


class DataProvider:
	SEED = 0  # random seed for validation set
	
	@property
	def data_shape(self):
		""" Return shape as python list of one data entry """
		raise NotImplementedError
	
	@property
	def n_classes(self):
		""" Return `int` of num classes """
		raise NotImplementedError
	
	@property
	def save_path(self):
		""" local path to save the data """
		raise NotImplementedError
	
	@property
	def data_url(self):
		""" link to download the data """
		raise NotImplementedError
	
	def labels_to_one_hot(self, labels):
		""" Convert 1D array of labels to one hot representation
		Args:
			labels: 1D numpy array
		"""
		new_labels = np.zeros((labels.shape[0], self.n_classes))
		new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
		return new_labels
	
	@staticmethod
	def labels_from_one_hot(labels):
		""" Convert 2D array of labels to 1D class based representation
		Args:
			labels: 2D numpy array
		"""
		return np.argmax(labels, axis=1)
