#!/usr/bin/env python3

import functools
import numpy
import sklearn.decomposition
import sklearn.discriminant_analysis
# custom lib
from . import base
import pylib.util


@base.DimReducerCollection.register("none", "plain")
@base.DimReducerAbstract.serialize_init(as_name = "none")
class Plain(base.DimReducerAbstract):
	"""
	'Plain' dimension reducer does not do any thing: it's a dummy class to hold
	the context consistency;
	"""
	def __init__(self, n_components = 0):
		# n_components is used to absorb the argument passed to initializer
		# it has no actual effect in this class
		pass
	def fit(self, X, *ka, **kw):
		self._d_in = X.shape[1]
		return self
	def transform(self, X, *ka, **kw):
		return X
	@property
	def feature_score(self):
		return numpy.ones(self._d_in, dtype = float)


@base.DimReducerCollection.register("lda", "linear_discriminant_analysis")
@base.DimReducerAbstract.serialize_init(as_name = "lda",
	params = ["n_components"])
class LinearDiscriminantAnalysis(
		sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
		base.DimReducerAbstract):
	def fit(self, X, y, *ka, **kw):
		# validate n_components
		self.n_components = min(self.n_components, len(set(y)) - 1)
		return super().fit(X, y, *ka, **kw)
	@property
	def feature_score(self):
		return self.scalings_[:, :35].T
		#return numpy.linalg.norm(self.scalings_, ord = 2, axis = 1)


@base.DimReducerCollection.register("pca", "principal_component_analysis")
@base.DimReducerAbstract.serialize_init(as_name = "pca",
	params = ["n_components"])
class PrincipalComponentAnalysis(sklearn.decomposition.PCA,
		base.DimReducerAbstract):
	@property
	def feature_score(self):
		return self.components_[:35].copy()
		#return numpy.linalg.norm(self.components_, ord = 2, axis = 0)


@base.DimReducerCollection.register("kpca", "kernel_principal_component_analysis")
@base.DimReducerAbstract.serialize_init(as_name = "kpca",
	params = ["n_components", "kernel", "gamma"])
class KernelPrincipalComponentAnalysis(sklearn.decomposition.KernelPCA,
		pylib.util.kernel_routines.RBFKernelRoutinesMixin,
		base.DimReducerAbstract):
	@functools.wraps(sklearn.decomposition.KernelPCA.__init__)
	def __init__(self, *ka, kernel = "rbf", **kw):
		# wraps for using rbf kernel as default
		super(KernelPrincipalComponentAnalysis, self)\
			.__init__(*ka, kernel = kernel, **kw)
		return

	@functools.wraps(sklearn.decomposition.KernelPCA.fit)
	def fit(self, X, *ka, use_default_gamma = True, **kw):
		if use_default_gamma:
			self.set_params(gamma = self.rbf_gamma_by_median(X))
		return super(KernelPrincipalComponentAnalysis, self)\
			.fit(X, *ka, **kw)
