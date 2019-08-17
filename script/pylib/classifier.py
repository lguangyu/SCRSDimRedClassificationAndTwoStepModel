#!/usr/bin/env python3

import functools
import numpy
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.svm
from . import base_class
from . import result_evaluate


_CLASSIFIERS = dict()


def register_classifier(registed_name):
	def inner_wrapper(cls):
		if issubclass(cls, base_class.ABCClassifier):
			cls.registed_name = registed_name
			_CLASSIFIERS.update({registed_name: cls})
			return cls
		raise TypeError("cls must be base_class.ABCClassifier")
	return inner_wrapper


class SklearnClassifierAliasMethodsMixin(base_class.ABCClassifier):
	def train(self, *ka, **kw):
		return super(SklearnClassifierAliasMethodsMixin, self).fit(*ka, **kw)

	def predict(self, *ka, **kw):
		return super(SklearnClassifierAliasMethodsMixin, self).predict(*ka, **kw)


@register_classifier("gnb")
class GNB(SklearnClassifierAliasMethodsMixin,\
	sklearn.naive_bayes.GaussianNB):
	@functools.wraps(sklearn.naive_bayes.GaussianNB.__init__)
	def __init__(self, **kw):
		super(GNB, self).__init__(**kw)
		return


@register_classifier("lr")
class LR(SklearnClassifierAliasMethodsMixin,\
	sklearn.linear_model.LogisticRegression):
	@functools.wraps(sklearn.linear_model.LogisticRegression.__init__)
	def __init__(self, **kw):
		super(LR, self).__init__(**kw)
		return


@register_classifier("lda")
class LDA(SklearnClassifierAliasMethodsMixin,\
	sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
	@functools.wraps(sklearn.discriminant_analysis.LinearDiscriminantAnalysis.__init__)
	def __init__(self, **kw):
		super(LDA, self).__init__(**kw)
		return


@register_classifier("svm_lin")
class SVMLinear(SklearnClassifierAliasMethodsMixin,\
	sklearn.svm.LinearSVC):
	# sklearn explicitly require the __init__ provide all argument names
	# here we bypass is by using wrapping
	@functools.wraps(sklearn.svm.LinearSVC.__init__)
	def __init__(self, **kw):
		super(SVMLinear, self).__init__(multi_class = "ovr", **kw)
		return


@register_classifier("svm_rbf")
class KSVM_RBF(SklearnClassifierAliasMethodsMixin,\
	sklearn.svm.SVC):
	@functools.wraps(sklearn.svm.SVC.__init__)
	def __init__(self, **kw):
		super(KSVM_RBF, self).__init__(kernel = "rbf", **kw)
		return

	def train(self, X, Y, *ka, **kw):
		# for rbf svm, gamma should be calculated
		sigma = numpy.median(sklearn.metrics.pairwise.euclidean_distances(X))
		gamma = 1.0 / (2 * sigma * sigma)
		self.set_params(gamma = gamma)
		assert self.get_params()["gamma"] == gamma
		return super(KSVM_RBF, self).train(X, Y, *ka, **kw)


@register_classifier("svm_lin_cv")
class SVMLinearCV(SVMLinear):
	def train(self, X, Y, *ka, **kw):
		c_list = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
		# inner loop to choose C
		best_c, best_acc = None, None
		cv = sklearn.model_selection.StratifiedKFold(n_splits = 5)
		for c in c_list:
			round_acc = list()
			for train, test in cv.split(X, Y):
				# train with
				self.set_params(C = c)
				super(SVMLinearCV, self).train(X[train], Y[train], *ka, **kw)
				pred = self.predict(X[test])
				acc = result_evaluate.LabelPredictEvaluate(pred_label = pred,
					true_label = Y[test])["average_accuracy"]
				round_acc.append(acc)
			# select best
			acc = numpy.mean(round_acc)
			if (best_acc is None) or (acc > best_acc):
				best_acc = acc
				best_c = c
		# retrain with best c
		self.set_params(C = c)
		super(SVMLinearCV, self).train(X, Y)
		return


@register_classifier("svm_rbf_cv")
class KSVM_RBF_CV(KSVM_RBF):
	def train(self, X, Y, *ka, **kw):
		c_list = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
		# inner loop to choose C
		best_c, best_acc = None, None
		cv = sklearn.model_selection.StratifiedKFold(n_splits = 5)
		for c in c_list:
			round_acc = list()
			for train, test in cv.split(X, Y):
				# train with
				self.set_params(C = c)
				super(KSVM_RBF_CV, self).train(X[train], Y[train], *ka, **kw)
				pred = self.predict(X[test])
				acc = result_evaluate.LabelPredictEvaluate(pred_label = pred,
					true_label = Y[test])["average_accuracy"]
				round_acc.append(acc)
			# select best
			acc = numpy.mean(round_acc)
			if (best_acc is None) or (acc > best_acc):
				best_acc = acc
				best_c = c
		# retrain with best c
		self.set_params(C = c)
		super(KSVM_RBF_CV, self).train(X, Y)
		return


def create(registed_name, *ka, **kw):
	if registed_name in _CLASSIFIERS:
		return _CLASSIFIERS[registed_name](*ka, **kw)
	raise RuntimeError("model must be one of: %s"\
		% repr(sorted(_CLASSIFIERS.keys())))


def list_registered():
	return sorted(_CLASSIFIERS.keys())
