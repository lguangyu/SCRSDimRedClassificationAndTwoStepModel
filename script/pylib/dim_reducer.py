#!/usr/bin/env python3

import functools
import numpy
import sklearn.decomposition
import sklearn.discriminant_analysis
from . import base_class
from . import HSIC_LSDR
from . import mixin


_DIM_REDUCERS = dict()


def register_dim_reducer(registed_name):
	def inner_wrapper(cls):
		if issubclass(cls, base_class.ABCDimensionReducer):
			cls.registed_name = registed_name
			_DIM_REDUCERS.update({registed_name: cls})
			return cls
		raise TypeError("cls must be base_class.ABCDimensionReducer")
	return inner_wrapper


class SklearnDimReducerAliasMethodsMixin(base_class.ABCDimensionReducer):
	def train(self, *ka, **kw):
		return super(SklearnDimReducerAliasMethodsMixin, self).fit(*ka, **kw)

	def transform(self, *ka, **kw):
		return super(SklearnDimReducerAliasMethodsMixin, self).transform(*ka, **kw)


@register_dim_reducer("none")
class Plain(base_class.ABCDimensionReducer):
	# dummy model does no dimension reduction
	def __init__(self, n_components = None, **kw):
		super(Plain, self).__init__(**kw)

	def train(self, X, Y):
		pass

	def transform(self, X):
		return X


@register_dim_reducer("pca")
class PCA(SklearnDimReducerAliasMethodsMixin,\
	sklearn.decomposition.PCA):
	@functools.wraps(sklearn.decomposition.PCA.__init__)
	def __init__(self, **kw):
		super(PCA, self).__init__(**kw)
		return


@register_dim_reducer("lda")
class LDA(SklearnDimReducerAliasMethodsMixin,\
	sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
	@functools.wraps(sklearn.discriminant_analysis.LinearDiscriminantAnalysis.__init__)
	def __init__(self, **kw):
		super(LDA, self).__init__(**kw)
		return


@register_dim_reducer("lsdr")
class LSDR(base_class.ABCDimensionReducer):
	# wrapper class for Chieh's LSDR
	# linear supervised dimension reduction using HSIC
	def __init__(self, n_components, **kw):
		# n_clusters can be acquired
		# when training data passed to fit()
		super(LSDR, self).__init__(**kw)
		if n_components < 0:
			raise ValueError("n_components must be postitive")
		self.reduce_dim_to = n_components
		return

	def _make_db(self, X, Y):
		# num of classes in labels (Y)
		self.num_of_clusters = len(numpy.unique(Y))
		# self.db is a local storage
		self.db = dict(
			X = X,
			Y = Y,
			num_of_clusters = self.num_of_clusters,
			q = self.reduce_dim_to,
			λ_ratio = 0.0,
			center_and_scale = False # we do it manually
		)
		return self.db

	def train(self, X, Y):
		db = self._make_db(X, Y)
		self.sdr = HSIC_LSDR.LSDR(db)
		self.sdr.train()
		return

	def transform(self, X):
		# on test dataset
		W = self.sdr.get_projection_matrix()
		# not scale the data, we do it outside
		return numpy.dot(X, W)


@register_dim_reducer("lsdr_reg")
class LSDR_REG(LSDR, mixin.RegularizerMixin):
	"""
	incorporated model selection (scale of regularization)
	"""
	def __init__(self, **kw):
		super(LSDR_REG, self).__init__(**kw)
		return

	def train(self, X, Y):
		db = self._make_db(X, Y)
		# update regularizer
		db["λ_ratio"] = self.regularizer
		self.sdr = HSIC_LSDR.LSDR(db)
		self.sdr.train()
		return

	def transform(self, X):
		return super(LSDR_REG, self).transform(X)


def create(registed_name, *ka, **kw):
	"""
	factory function of dimension reduction family objects;
	returned object must support fit(), transform()
	and fit_transform() as interface;
	thus the caller does not have to know what types of models we have;
	all such information is organized here.
	"""
	if registed_name in _DIM_REDUCERS:
		return _DIM_REDUCERS[registed_name](*ka, **kw)
	raise RuntimeError("model must be one of: %s"\
		% repr(sorted(_DIM_REDUCERS.keys())))


def list_registered():
	return sorted(_DIM_REDUCERS.keys())
