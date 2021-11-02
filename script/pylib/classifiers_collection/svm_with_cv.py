#!/usr/bin/env python3

import numpy
# custom lib
from . import base
from . import simple
from . import cv_parameter_selector


@base.ClassifierCollection.register("svm_lin_cv", "linear_svm_cv")
@base.ClassifierAbstract.serialize_init(as_name = "svm_lin_cv", params = ["C"])
class LinearSVM_CV(cv_parameter_selector._CVClassifParamSelectMixin,
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
class RBFKernelSVM_CV(cv_parameter_selector._CVClassifParamSelectMixin,
		simple.RBFKernelSVM):
	"""
	rbf kernel svm using cross validation to select both the best penalty C and
	the scaling parameter gamma;
	"""
	def fit(self, X, Y, *ka, **kw):
		_sp = numpy.linspace(-5, 5, 11)
		pars = dict(C = numpy.power(10, _sp),
			gamma = numpy.power(2, _sp) * self.rbf_gamma_by_median(X))
		return super(RBFKernelSVM_CV, self).fit(X, Y, *ka,
			use_default_gamma = False, cv_params = pars, **kw)
