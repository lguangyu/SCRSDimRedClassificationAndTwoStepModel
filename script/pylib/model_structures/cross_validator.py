#!/usr/bin/env python3

import abc
import multiprocessing
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

	def add_cv_results(self, *result):
		self.__cv_results.extend(result)
		return

	def _cv_thread_arg_iter(self, cv_obj, model, X, Y, *, duo_label = False):
		for train, test in cv_obj.split(X, Y[1] if duo_label else Y):
			Y_train	= [y[train] for y in Y] if duo_label else Y[train]
			Y_test	= Y[1][test] if duo_label else Y[test]
			# model, x_train, y_train, x_test, y_test
			yield model, X[train], Y_train, X[test], Y_test
		return

	@staticmethod
	def _cv_thread_func(model, X_train, Y_train, X_test, Y_test):
		# use force_create = True will maximize the resource release
		model.fit(X_train, Y_train, force_create = True)
		model.predict(X_test, Y_test)
		ret = model.serialize()
		return ret


	def cross_validate(self, model, X, Y, *ka, duo_label = False, n_jobs = 1,
			**kw):
		if not isinstance(model, base.ModelStructureAbstract):
			raise TypeError("model must be ModelStructureAbstract, not '%s'"\
				% type(model).__name__)
		self.reset_cv_results() # remove old results
		cv = self.cv_driver(**self.cv_props)
		pool = multiprocessing.Pool(n_jobs)
		res = pool.starmap(self._cv_thread_func,
			self._cv_thread_arg_iter(cv, model, X, Y, duo_label = duo_label))
		self.add_cv_results(*res)
		#for train, test in cv.split(X, Y[1] if duo_label else Y):
		#	Y_train	= [y[train] for y in Y] if duo_label else Y[train]
		#	Y_test	= Y[1][test] if duo_label else Y[test]
		#	# use force_create = True will maximize the resource release
		#	model.fit(X[train], Y_train, force_create = True)
		#	model.predict(X[test], Y_test)
		#	self.add_cv_results(model.serialize())
		return self
