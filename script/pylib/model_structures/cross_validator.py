#!/usr/bin/env python3

import abc
import sklearn.model_selection
# custom lib
import pylib.util
from . import base


class CrossValidator(object):
	"""
	evaluate model performance using cross validation; the input dataset will be
	splitted into even <n> splits to feed independent model runs; for each run,
	one split is set as testing set and the rest are training set; performance
	evaluation is mainly based on the testing set, while the training evaluation
	will be kept as a reference;
	"""
	def __init__(self, cv_driver = sklearn.model_selection.StratifiedKFold,
			cv_props = dict(), *ka, **kw):
		"""
		create a cross-validation framework using <cv_driver> as cross validator
		and <cv_props> to initialze the cross validator;
		"""
		super(CrossValidator, self).__init__(*ka, **kw)
		self.cv_driver = cv_driver
		self.cv_props = cv_props
		self.reset_cv_results()
		return

	def reset_cv_results(self):
		self.__cv_results = list()

	def get_cv_results(self):
		return self.__cv_results.copy()

	def add_cv_results(self, result):
		self.__cv_results.append(result)
		return

	def cross_validate(self, model, X, Y, *ka, **kw):
		if not isinstance(model, base.ModelStructureAbstract):
			raise TypeError("model must be ModelStructureAbstract, not '%s'"\
				% type(model).__name__)
		self.reset_cv_results() # remove old results
		cv = self.cv_driver(**self.cv_props)
		for train, test in cv.split(X, Y):
			model.fit(X[train], Y[train])
			model.predict(X[test], Y[test])
			self.add_cv_results(model.serialize())
		return self
