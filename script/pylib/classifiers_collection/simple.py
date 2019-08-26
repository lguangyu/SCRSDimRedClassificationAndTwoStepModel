#!/usr/bin/env python3

import functools
import numpy
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.svm
# custom lib
from . import base
import pylib.util


@base.ClassifierCollection.register("gnb", "gaussian_naive_bayesian")
@base.ClassifierAbstract.serialize_init(as_name = "gnb")
class GaussianNaiveBayesian(sklearn.naive_bayes.GaussianNB,
		base.ClassifierAbstract):
	pass


@base.ClassifierCollection.register("lr", "logistic_regression")
@base.ClassifierAbstract.serialize_init(as_name = "lr")
class LogisticRegression(sklearn.linear_model.LogisticRegression,
		base.ClassifierAbstract):
	# ok.. we have to do this
	@functools.wraps(sklearn.linear_model.LogisticRegression.predict)
	def predict(self, *ka, **kw):
		return sklearn.linear_model.LogisticRegression.predict(self, *ka, **kw)


@base.ClassifierCollection.register("lda", "linear_discriminant_analysis")
@base.ClassifierAbstract.serialize_init(as_name = "lda")
class LinearDiscriminantAnalysis(
		sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
		base.ClassifierAbstract):
	# ok.. we have to do this
	@functools.wraps(sklearn.discriminant_analysis.LinearDiscriminantAnalysis\
		.predict)
	def predict(self, *ka, **kw):
		return sklearn.discriminant_analysis.LinearDiscriminantAnalysis\
			.predict(self, *ka, **kw)


@base.ClassifierCollection.register("svm_lin", "linear_svm")
@base.ClassifierAbstract.serialize_init(as_name = "svm_lin")
class LinearSVM(sklearn.svm.LinearSVC, base.ClassifierAbstract):
	@functools.wraps(sklearn.svm.LinearSVC.__init__)
	def __init__(self, **kw):
		super(LinearSVM, self).__init__(multi_class = "ovr", **kw)
		return
	# ok.. we have to do this
	@functools.wraps(sklearn.svm.LinearSVC.predict)
	def predict(self, *ka, **kw):
		return sklearn.svm.LinearSVC.predict(self, *ka, **kw)


@base.ClassifierCollection.register("svm_rbf", "rbf_kernel_svm")
@base.ClassifierAbstract.serialize_init(as_name = "svm_rbf",
	params = ["gamma"])
class RBFKernelSVM(sklearn.svm.SVC,
		pylib.util.kernel_routines.RBFKernelRoutinesMixin,
		base.ClassifierAbstract):
	"""
	kernel support vector machine using rbf (gaussian) kernel; default scaling
	parameter is calculated as sigma = <median euclidean distances>, if
	default_gamma = True (default) passed to .fit();
	method;
	"""
	@functools.wraps(sklearn.svm.SVC.__init__)
	def __init__(self, **kw):
		super(RBFKernelSVM, self).__init__(kernel = "rbf", **kw)
		return

	def fit(self, X, Y, *ka, use_default_gamma = True, **kw):
		if use_default_gamma:
			# calculate gamma if not provided, by default the median of pairwise
			# euclidean distances
			self.set_params(gamma = self.rbf_gamma_by_median(X))
		return super(RBFKernelSVM, self).fit(X, Y, *ka, **kw)
