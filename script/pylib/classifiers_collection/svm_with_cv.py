#!/usr/bin/env python3

import functools
import gc
import numpy
import sklearn.metrics
import sys
# custom lib
from . import base
from . import simple
from . import cv_parameter_selector


@base.ClassifierCollection.register("svm_lin_cv", "linear_svm_cv")
@base.ClassifierAbstract.serialize_init(as_name = "svm_lin_cv", params = ["C"])
class LinearSVM_CV(cv_parameter_selector.CVClassifParamSelectMixin,
		simple.LinearSVM):
	"""
	linear svm using cross validation to select the best penalty parameter, C
	"""
	def fit(self, X, Y, *ka, **kw):
		# set parameters to cv
		_sp = numpy.linspace(-5, 5, 11)
		pars = dict(C = numpy.power(10, _sp))
		return super(LinearSVM_CV, self).fit(X, Y, *ka, cv_params = pars, **kw)


@base.ClassifierCollection.register("svm_rbf_cv", "rbf_kernel_svm_cv")
@base.ClassifierAbstract.serialize_init(as_name = "svm_rbf_cv",
	params = ["C", "gamma"])
class RBFKernelSVM_CV(cv_parameter_selector.CVClassifParamSelectMixin,
		simple.RBFKernelSVM):
	"""
	rbf kernel svm using cross validation to select both the best penalty C and
	the scaling parameter gamma;
	"""
	class RBFKernelMatrixCache(object):
		def __init__(self, *ka, **kw):
			super().__init__(*ka, **kw)
			self.X			= None
			self.gamma		= None
			self.gram_mat	= None
			return

		def attempt_get_gram_mat(self, X, gamma):
			# re-calculate, cache (and return) the gram matrix if X or gamma
			# does not match; otherwise direct return the cached matrix
			if (id(X) != id(self.X)) or (gamma != self.gamma):
				self.X			= X
				self.gamma		= gamma
				self.gram_mat	= sklearn.metrics.pairwise_kernels(X,
					metric = "rbf", gamma = gamma)
			#	print("recalculate kernel", file = sys.stderr)
			#else:
			#	print("using existing kernel")
			# run gc to release the old gram_mat memory
			gc.collect()
			return self.gram_mat

	def get_gram_mat_by_gamma(self, X, gamma):
		"""
		get the gram matrix based on X and gamma; use cached matrix if possible;
		"""
		if not hasattr(self, "_gram_mat_cache"):
			self._gram_mat_cache = type(self).RBFKernelMatrixCache()
		return self._gram_mat_cache.attempt_get_gram_mat(X, gamma)

	def param_fit_predict(self, X, Y, train, test, *ka, param, **kw):
		"""
		overrides the function in CVClassifParamSelectMixin by using
		pre-calculated kernels in self._gram_mats
		"""
		self.set_params(**param, kernel = "precomputed")
		gram_mat = self.get_gram_mat_by_gamma(X, param["gamma"])
		# now we need to figure out X, Y in precomputed self._gram_mats
		K_train = gram_mat[numpy.ix_(train, train)]
		Y_train = Y[train]
		# here we need to skip a bit more then just using super().fit()
		super(cv_parameter_selector.CVClassifParamSelectMixin, self)\
			.fit(K_train, Y_train, *ka, **kw)
		# similar for predict() call
		K_test = gram_mat[numpy.ix_(test, train)]
		pred = super(cv_parameter_selector.CVClassifParamSelectMixin, self)\
			.predict(K_test)
		return pred

	def fit(self, X, Y, *ka, **kw):
		_sp = numpy.linspace(-5, 5, 7)
		pars = dict(C = numpy.power(10, _sp),
			gamma = numpy.power(3, _sp) * self.rbf_gamma_by_median(X))
			# use median Euc distance as reference, low = 2^-5, high = 2^5
		return super(RBFKernelSVM_CV, self).fit(X, Y, *ka,
			use_default_gamma = False, cv_params = pars, **kw)

	def post_param_cv(self):
		# during parameter selection, the kernel is temporarily set to
		# 'precomputed' to avoid repetitive calculations; now its time to set
		# it back to rbf
		self.set_params(kernel = "rbf")
		return

	def sort_candidate_params_list(self, params_list):
		# sort by gamma; so we can use the cached gram_mat as much as possible
		# to prevent repetitive computation
		return sorted(params_list, key = lambda x: x["gamma"])
