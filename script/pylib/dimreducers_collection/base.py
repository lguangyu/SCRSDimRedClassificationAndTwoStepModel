#!/usr/bin/env python3

import abc
import sklearn.base
# custom lib
import pylib.util


class DimReducerAbstract(pylib.util.model_serializer.ModelSerializerBase,
		sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, abc.ABC):
	"""
	classifier abstract base class; all classifier must subclass this base;
	"""
	@abc.abstractmethod
	def fit(self, *ka, **kw):
		return self
	@abc.abstractmethod
	def transform(self, *ka, **kw):
		pass


@pylib.util.model_collection.ModelCollection.init(DimReducerAbstract)
class DimReducerCollection(pylib.util.model_collection.ModelCollection):
	"""
	collection registry of all dimension reducer classes; all elements should
	subclass DimReducerAbstract to be recogized as valid;
	"""
	pass
