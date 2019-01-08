#!/usr/bin/env python3

import sklearn.decomposition
import sklearn.discriminant_analysis


class DimensionReductionBase(object):
	def __init__(self, *ka, **kw):
		super(DimensionReductionBase, self).__init__()

	def fit(self, *ka, **kw):
		raise NotImplementedError
		return

	def transform(self, *ka, **kw):
		raise NotImplementedError()
		return

	def fit_transform(self, *ka, **kw):
		raise NotImplementedError()
		return


class Plain(DimensionReductionBase):
	# plain model does no dimension reduction
	# just a fitting class
	def __init__(self, *ka, **kw):
		super(Plain, self).__init__(*ka, **kw)

	def fit(self, X, Y):
		pass

	def transform(self, X):
		return X

	def fit_transform(self, X, Y):
		#self.fit(X)
		return X


class PCA(sklearn.decomposition.PCA,
	DimensionReductionBase):
	# DimensionReductionBase is the later one
	# use sklearn.decomposition.PCA methods in prior
	def __init__(self, *ka, **kw):
		super(PCA, self).__init__(*ka, **kw)


class LDA(sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
	DimensionReductionBase):
	# same to PCA
	def __init__(self, *ka, **kw):
		super(LDA, self).__init__(*ka, **kw)


class LSDR(DimensionReductionBase):
	# wrapper class for Chieh's LSDR
	# linear supervised dimension reduction using HSIC
	def __init__(self, reduce_dim_to, *ka, **kw):
		# n_clusters can be acquired
		# when training data passed to fit()
		super(LSDR, self).__init__(*ka, **kw)


#	def fit(self, X, Y):
#		pass
#
#
#	def transform(self, X):
#		pass
#
#
#	def fit_transform(self, X, Y):
#		pass


def get_dim_reduction_object(model, reduce_dim_to):
	if reduce_dim_to <= 0:
		raise ValueError("reduce_dim_to must be positive")
	if model is None:
		return Plain()
	elif model == "pca":
		return PCA(n_components = reduce_dim_to)
	elif model == "lda":
		return LDA(n_components = reduce_dim_to)
	elif model == "lsdr":
		return LSDR(reduce_dim_to = reduce_dim_to)
	else:
		raise ValueError("unrecognized model '%s'" % model)
