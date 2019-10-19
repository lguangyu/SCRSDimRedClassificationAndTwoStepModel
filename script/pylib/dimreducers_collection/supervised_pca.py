#!/usr/bin/env python3
################################################################################
# this module implements the supervised pca algorithm [1]
#
# REFERENCES:
# [1] Barshan et al. (2011) "Supervised principal component analysis:
#   Visualization, classification and regression on subspaces and submanifolds"
################################################################################

import functools
import numpy
import scipy.sparse
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.preprocessing
import sklearn.utils
# custom lib
from . import base
import pylib.util


@base.DimReducerCollection.register("sup_pca",
	"supervised_principal_component_analysis")
@base.DimReducerAbstract.serialize_init(as_name = "sup_pca",
	params = ["n_components"])
class SupervisedPrincipalComponentAnalysis(
		pylib.util.kernel_routines.RBFKernelRoutinesMixin, # for extension
		base.DimReducerAbstract):
	def __init__(self, *, n_components, tol = 0.0):
		self.n_components = n_components
		self.tol = tol
		return

	def _one_hot_encode_y(self, Y):
		self._one_hot_enc = sklearn.preprocessing.OneHotEncoder(sparse = False)
		enc_y = self._one_hot_enc.fit_transform(Y.reshape(-1, 1))
		return enc_y

	def _get_cov_matrix(self, X, Y):
		XH = self.centering(X, copy = True) # from kernel routines class
		enc_y = self._one_hot_encode_y(Y)
		# for extension to supervised kernel pca,
		# replace L with corr. kernel matrix
		L = numpy.dot(enc_y, enc_y.T) # this is linear kernel
		cov = numpy.linalg.multi_dot((XH.T, L, XH))
		return cov

	def fit(self, X, Y):
		if X.ndim != 2:
			raise ValueError("X must be 2-d, got %d-d" % X.ndim)
		if len(X) != len(Y):
			raise ValueError("X and Y must contain same numbe of samples")
		# shape of X?
		n_samples, n_features = X.shape
		n_components = min(n_samples, n_features, self.n_components)
		# svd
		cov = self._get_cov_matrix(X, Y)
		# below follows sklearn.decomposition.pca.PCA._fit_truncated() routine
		U, S, V = scipy.sparse.linalg.svds(cov, k = n_components,
			tol = self.tol)
		S = S[::-1]
		U, V = sklearn.utils.extmath.svd_flip(U, V)
		# result summary
		self.n_samples_, self.n_features_ = n_samples, n_features
		self.n_components_ = n_components
		self.components_ = V
		self.singular_values_ = S.copy()
		return self

	def transform(self, X):
		if X.ndim != 2:
			raise ValueError("X must be 2-d, got %d-d" % X.ndim)
		ret = numpy.dot(X, self.components_.T)
		return ret
