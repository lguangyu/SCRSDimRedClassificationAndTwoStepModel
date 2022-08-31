#!/usr/bin/env python3

import abc
# custom lib
import pylib.util


class FeatureRankAbstract(object):
	"""
	base class for feature ranking algorithms
	"""
	@abc.abstractmethod
	def rank_features(self, X, Y = None):
		"""
		return an array of feature indices, arranged in descending order
		of feature *importance*
		"""
		pass


@pylib.util.collection_registry.CollectionRegistryBase.init(FeatureRankAbstract)
class FeatureRankCollection(
		pylib.util.collection_registry.CollectionRegistryBase):
	"""
	collection registry of all feature ranking algorithm classes; all elements
	should subclass FeatureRankAbstract to be recogized as valid;
	"""
	pass
