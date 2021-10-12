#!/usr/bin/env python3

import sklearn.base
# custom lib
import pylib.util


class ClassifierAbstract(pylib.util.model_serializer.ModelSerializerBase,
		sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
	"""
	classifier abstract base class; all classifier must subclass this base;
	"""
	pass


@pylib.util.model_collection.ModelCollection.init(ClassifierAbstract)
class ClassifierCollection(pylib.util.model_collection.ModelCollection):
	"""
	collection registry of all classifier classes; all elements should subclass
	ClassifierAbstract to be recogized as valid;
	"""
	pass
