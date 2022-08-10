#!/usr/bin/env python3

import sklearn.base
# custom lib
import pylib.util


class DimReducerAbstract(pylib.util.model_serializer.ModelSerializerBase,
		sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
	"""
	classifier abstract base class; all classifier must subclass this base;
	"""
	@property
	def feature_score(self):
		"""
		report the scores of each input feature; if used on a DR class that
		does not implement this method, a NotImplementedError will be raised
		"""
		raise NotImplementedError("feature_score not implemented by '%s'"\
			% type(self).__name__)


@pylib.util.model_collection.ModelCollection.init(DimReducerAbstract)
class DimReducerCollection(pylib.util.model_collection.ModelCollection):
	"""
	collection registry of all dimension reducer classes; all elements should
	subclass DimReducerAbstract to be recogized as valid;
	"""
	pass
