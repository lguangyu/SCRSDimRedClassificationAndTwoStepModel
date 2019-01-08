#!/usr/bin/env python3

import numpy
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.metrics # used to calculate rbf kernel
import sklearn.discriminant_analysis
import sklearn.svm


# custom classifiers
# should support at least fit() and predict() methods

def get_classifier_object(model, data = None):
	"""
	factory function to get classifier objects
	'data' is only required by svm_rbf
	"""
	if model == "gnb":
		return sklearn.naive_bayes.GaussianNB()
	elif model == "lr":
		return sklearn.linear_model.LogisticRegression()
	elif model == "lda":
		return sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
	elif model == "svm_lin":
		return sklearn.svm.LinearSVC(multi_class = "ovr")
	elif model == "svm_rbf":
		if data is None:
			raise ValueError("'data' is required for 'svm_rbf'")
		sigma = numpy.median(sklearn.metrics.pairwise.euclidean_distances(data))
		gamma = 1.0 / (2 * sigma * sigma)
		return sklearn.svm.SVC(kernel = "rbf", gamma = gamma)
	else:
		raise RuntimeError("unrecognized model '%s'" % model)
	return
