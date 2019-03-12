#!/usr/bin/env python3

import numpy


class EvenSplitCrossValidation(object):
	def __init__(self, data, labels, n_fold = 10,
		permutation = "disable"):
		"""
		generate dataset splits for cross validation;
		labels must be encoded;
		permutation can be: 'disable', 'random', integer or an array
		if 'random', randomly generate the permutation table
		if integer, using it as seed and generate the permutation table
		if array, use as permutation table
		"""
		super(EvenSplitCrossValidation, self).__init__()
		# ensure copy below two as numpy array
		self.data = numpy.array(data)
		self.labels = numpy.array(labels, dtype = int)
		# self.permutation_table = None if disabled
		# otherwise the table either passed in or generated
		self.permutation_table = self._permutate_inplace(permutation)
		self.n_fold = n_fold
		self._test_masks = \
			self._generate_test_masks(self.data, self.labels, self.n_fold)


	def _permutate_inplace(self, permutation):
		if permutation == "disable":
			# do nothing
			return None
		else:
			# get permutation table
			if permutation == "random":
				_table = numpy.random.permutation(len(self.labels))
			elif isinstance(permutation, int):
				numpy.random.seed(permutation)
				_table = numpy.random.permutation(len(self.labels))
			else:
				_table = permutation.copy()
			# permutate inplace
			self.data	= self.data[_table]
			self.labels	= self.labels[_table]
			return _table
		return #


	@staticmethod
	def _get_label_split_mask(labels_vec, target_label, n_splits):
		"""
		split the positions of <target_label> in <labels_vec> into
		almost-evenly distributed <n_splits> query indices
		returns a list of indices arrays
		"""
		# nonzero indices
		target_label_mask = numpy.equal(labels_vec, target_label)
		indices = numpy.nonzero(target_label_mask)[0]
		# this function deals with n-elements not dividable by n_splits
		folds_masks = numpy.array_split(indices, n_splits)
		assert type(folds_masks) == list
		return folds_masks


	@staticmethod
	def _generate_test_masks(data, labels, n_fold):
		# find each label mask (bool vector)
		# then split each label mask by _split_bool_vector
		# combine corresponding split 
		# get unique labels
		uniq_labels = numpy.unique(labels)
		uniq_labels.sort()
		# find splits for each label
		label_splits = [EvenSplitCrossValidation.\
			_get_label_split_mask(labels, i, n_fold)
			for i in uniq_labels]
		# note these are indices,
		# next, each split just need to assign these indices to True
		# in a originally full of False vector
		ret = []
		for index in range(n_fold):
			mask = numpy.full(len(labels), False, dtype = bool)
			for _sp in label_splits:
				# structure: label[#labels][#splits]
				_indices = _sp[index]
				mask[_indices] = True
			ret.append(mask)
		return ret


	def get_split(self, index, return_mask = False):
		"""
		if return_mask is False, return train/test datasets (X and Y)
		else return the corresponding masks
		"""
		test_mask = self._test_masks[index]
		train_mask = numpy.logical_not(test_mask)
		if return_mask:
			# return only two
			return train_mask, test_mask
		else:
			train_data = self.data[train_mask]
			test_data = self.data[test_mask]
			train_label = self.labels[train_mask]
			test_label = self.labels[test_mask]
			# need to return four
			return train_data, test_data, train_label, test_label


	def __getitem__(self, index):
		"""
		alias to self.get_split(index, return_mask = False)
		"""
		return self.get_split(index, return_mask = False)


	def __next__(self):
		"""
		enables using of 'for tX, vX, tY, vY in escv: ...'
		iterate through all splits
		"""
		i = 0
		while i < self.n_fold:
			yield self[i]
