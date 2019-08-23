#!/usr/bin/env python3

import sklearn.decomposition
import sklearn.discriminant_analysis
# custom lib
from . import base


@base.DimReducerCollection.register("none", "plain")
@base.DimReducerAbstract.serialize_init(as_name = "none")
class Plain(base.DimReducerAbstract):
	"""
	'Plain' dimension reducer does not do any thing: its a dummy class to hold
	the consistency in context;
	"""
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
