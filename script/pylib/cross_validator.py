#!/usr/bin/env python3

import os
import sklearn.model_selection
import sys
from . import base_class


class SingleLevelCrossValidator(object):
	############################################################################
	# variables
	@property
	def model(self):
		return self._model
	@model.setter
	def model(self, value):
		if isinstance(value, base_class.ABCModel):
			self._model = value
			return
		raise TypeError("model must be instance of model.ABCModel")

	@property
	def n_fold(self):
		return self._n_fold
	@n_fold.setter
	def n_fold(self, value):
		if value >= 2:
			self._n_fold = value
			return
		raise ValueError("n_fold must be at least 2")

	@property
	def permutation(self):
		return self._permutation
	@permutation.setter
	def permutation(self, value):
		if value in ["disable", "random"]:
			self._permutation = value
			return
		else:
			try:
				self._permutation = int(value)
				return
			except ValueError as e:
				pass
		raise ValueError("permutation must be disable, random or an int")

	# these are two related arguments derived from self.permutation
	@property
	def use_shuffle(self):
		return self.permutation != "disable"

	@property
	def shuffle_seed(self):
		return self.permutation if isinstance(self.permutation, int) else None

	@property
	def evaluation(self):
		return self._evaluation

	############################################################################
	# methods
	def __init__(self, model, n_fold = 10, permutation = "disable", *ka, **kw):
		super(SingleLevelCrossValidator, self).__init__(*ka, **kw)
		self.model = model
		self.n_fold = n_fold
		self.permutation = permutation
		self._evaluation = []
		return

	def run_cv(self, X, Y):
		"""
		split dataset then run cross validation

		assumes data (X) are preprocessed (e.g. scaled)
		assumes labels (Y) are encoded
		"""
		# clear old results
		self.evaluation.clear()

		cv_splitter = sklearn.model_selection.StratifiedKFold(\
			n_splits = self.n_fold,
			shuffle = self.use_shuffle,
			random_state = self.shuffle_seed)
		for train_indices, test_indices in cv_splitter.split(X, Y):
			# split
			# FIXME: this is temporary
			print("CV")
			train_X = X[train_indices]
			train_Y = Y[train_indices]
			test_X = X[test_indices]
			test_Y = Y[test_indices]
			# train and test
			self.model.train(train_X, train_Y)
			self.model.test(test_X, test_Y)
			# fetch results
			self.evaluation.append(self.model.evaluation.copy())
		return


class TwoLevelCrossValidator(SingleLevelCrossValidator):
	def __init__(self, *ka, **kw):
		super(TwoLevelCrossValidator, self).__init__(*ka, **kw)
		return

	def run_cv(self, X, univ_Y, level1_Y, level2_Y):
		"""
		split dataset then run cross validation

		assumes data (X) are preprocessed (e.g. scaled)
		assumes labels (Y) are encoded

		univ_Y: used to split folds (split both level1_Y and level2_Y)
		level1_Y, level2_Y: used in training
		"""
		# clear old results
		self.evaluation.clear()

		cv_splitter = sklearn.model_selection.StratifiedKFold(\
			n_splits = self.n_fold,
			shuffle = self.use_shuffle,
			random_state = self.shuffle_seed)
		for train_indices, test_indices in cv_splitter.split(X, univ_Y):
			# split
			train_X = X[train_indices]
			train_level1_Y = level1_Y[train_indices]
			train_level2_Y = level2_Y[train_indices]
			test_X = X[test_indices]
			test_level1_Y = level1_Y[test_indices]
			test_level2_Y = level2_Y[test_indices]
			# train and test
			self.model.train(train_X, train_level1_Y, train_level2_Y)
			self.model.test(test_X, test_level1_Y, test_level2_Y)
			# fetch results
			self.evaluation.append(self.model.evaluation.copy())
		return
