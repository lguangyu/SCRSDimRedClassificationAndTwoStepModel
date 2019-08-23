#!/usr/bin/env python3

import abc
import sklearn.base
# custom lib
import pylib.util


class ClassifierAbstract(pylib.util.model_serializer.ModelSerializerBase,
		sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin, abc.ABC):
	"""
	classifier abstract base class; all classifier must subclass this base;
	"""
	@abc.abstractmethod
	def fit(self, *ka, **kw):
		return self
	@abc.abstractmethod
	def predict(self, *ka, **kw):
		pass


@pylib.util.model_collection.ModelCollection.init(ClassifierAbstract)
class ClassifierCollection(pylib.util.model_collection.ModelCollection):
	"""
	collection registry of all classifier classes; all elements should subclass
	ClassifierAbstract to be recogized as valid;
	"""
	pass
