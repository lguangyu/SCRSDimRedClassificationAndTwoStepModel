#!/usr/bin/env python3

import numpy


class EvenSplitCrossValidation(object):
	def __init__(self, data, labels, n_fold = 10,
		permutation = False, permutation_table = None):
		"""
		generate dataset splits for cross validation;
		labels must be encoded;
		"""
		super(EvenSplitCrossValidation, self).__init__()
		# ensure copy below two as numpy array
		self.data = numpy.array(data)
		self.labels = numpy.array(labels, dtype = int)
		self.permutation_table =\
			self._permutate(permutation, permutation_table)
		self.n_fold = n_fold
		self._test_masks = \
			self._generate_test_masks(self.data, self.labels, self.n_fold)


	def _permutate(self, permutation, permutation_table):
		"""
		only permutate data if permutation = True;
		if so, if permutation_table is None, randomly generate;
		else, use specified one
		"""
		if permutation:
			if permutation_table is None:
				permutation_table = numpy.random.permutation(len(self.labels))
			# permutate both data and labels
			self.data	= self.data[permu_table]
			self.labels	= self.labels[permu_table]
			return permutation_table
		else:
			return None


	@staticmethod
	def _split_bool_vector(bool_vec, n_splits):
		"""
		split the True's in a boolean vector into n_splits;
		each split has almost same True's
		return list of indices
		"""
		# nonzero indices
		nonzero = (numpy.nonzero(bool_vec)[0]).tolist() # turn to list
		slice_size = len(nonzero) / n_splits # this is ok to be decimal
		ret = []
		for i in range(n_splits):
			slice_start	= int(slice_size * i)
			slice_end	= int(slice_size * (i + 1))
			ret.append(nonzero[slice_start:slice_end])
		return ret


	@staticmethod
	def _generate_test_masks(data, labels, n_fold):
		# find each label mask (bool vector)
		# then split each label mask
		# combine corresponding split 
		# get unique labels
		uniq_labels = numpy.unique(labels)
		uniq_labels.sort()
		# find splits for each label
		label_splits = [EvenSplitCrossValidation.\
			_split_bool_vector(labels == label, n_fold)
			for label in uniq_labels]
		#
		ret = []
		for index in range(n_fold):
			mask = numpy.full(len(labels), False, dtype = bool)
			for _sp in label_splits:
				# label[#labels][#splits]
				_indices = _sp[index]
				mask[_indices] = True
			ret.append(mask)
		return ret


	def get_split(self, index, return_mask = False):
		"""
		if return_mask is False, return train/test datasets
		else return the corresponding masks
		"""
		test_mask = self._test_masks[index]
		train_mask = numpy.logical_not(test_mask)
		if return_mask:
			return train_mask, test_mask
		else:
			train_data = self.data[train_mask]
			test_data = self.data[test_mask]
			train_label = self.labels[train_mask]
			test_label = self.labels[test_mask]
			return train_data, test_data, train_label, test_label


	def __getitem__(self, index):
		"""
		alias to self.get_split(index, return_mask = False)
		"""
		return self.get_split(index, return_mask = False)


	def __next__(self):
		i = 0
		while i < self.n_fold:
			yield self[i]
