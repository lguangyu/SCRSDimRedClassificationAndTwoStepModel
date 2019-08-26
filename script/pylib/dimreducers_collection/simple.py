#!/usr/bin/env python3

import functools
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

	def fit(self, *ka, **kw):
		return self

	def transform(self, X, *ka, **kw):
		return X


@base.DimReducerCollection.register("lda", "linear_discriminant_analysis")
@base.DimReducerAbstract.serialize_init(as_name = "lda",
	params = ["n_components"])
class LinearDiscriminantAnalysis(
		sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
		base.DimReducerAbstract):
	pass


@base.DimReducerCollection.register("pca", "principal_component_analysis")
@base.DimReducerAbstract.serialize_init(as_name = "pca",
	params = ["n_components"])
class PrincipalComponentAnalysis(sklearn.decomposition.PCA,
		base.DimReducerAbstract):
	pass


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
