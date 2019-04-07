#!/usr/bin/env python3

import collections
from . import base_class
from . import classifier as classifier_m
from . import dim_reducer as dim_reducer_m


class LevelPropsMixin(object):
	"""
	manager class of dim_reducer/classifier used in each level
	used as arguments in constructing LevelBasedModelBase() instances
	"""
	############################################################################
	# variables
	@property
	def dims_remain(self):
		return self._dims_remain
	@dims_remain.setter
	def dims_remain(self, value):
		if (value is None) or (value > 0):
			self._dims_remain = value
			return
		raise ValueError("dims after reduction must be positive")

	@property
	def classifier(self):
		return self._classifier
	@classifier.setter
	def classifier(self, value):
		self._classifier = value
		return
	
	@property
	def dim_reducer(self):
		return self._dim_reducer
	@dim_reducer.setter
	def dim_reducer(self, value):
		self._dim_reducer = value
		return

	@property
	def classifier_obj(self):
		"""
		get the instance of classifier
		"""
		if not hasattr(self, "_classifier_obj"):
			self.new_classifier_obj()
		return self._classifier_obj

	@property
	def dim_reducer_obj(self):
		"""
		get the instance of classifier
		"""
		if not hasattr(self, "_dim_reducer_obj"):
			self.new_dim_reducer_obj()
		return self._dim_reducer_obj

	############################################################################
	# methods
	def __init__(self, *ka, dim_reducer, classifier, dims_remain = None, **kw):
		super(LevelPropsMixin, self).__init__(*ka, **kw)
		self.dim_reducer = dim_reducer
		self.dims_remain = dims_remain
		self.classifier = classifier
		return

	def _get_classifier_obj(self, **kw):
		return classifier_m.create(self.classifier, **kw)

	def _get_dim_reducer_obj(self, **kw):
		return dim_reducer_m.create(self.dim_reducer,\
			n_components = self.dims_remain, **kw)

	def new_classifier_obj(self, **kw):
		self._classifier_obj = self._get_classifier_obj(**kw)
		return self._classifier_obj

	def new_dim_reducer_obj(self, **kw):
		self._dim_reducer_obj = self._get_dim_reducer_obj(**kw)
		return self._dim_reducer_obj

	def duplicate(self, astype = None):
		"""
		clone self settings to a new LevelPropsMixin instance
		"""
		if astype is None:
			astype = type(self)
		if not issubclass(astype, LevelPropsMixin):
			raise TypeError("astype must be subclass of LevelPropsMixin")
		# copy
		new = astype(\
			dim_reducer = self.dim_reducer,\
			classifier = self.classifier,\
			dims_remain = self.dims_remain)
		return new


class LevelBasedModelBase(base_class.ABCModel):
	"""
	model base class which can be separated into levels (ok if #levels = 1)
	in each level, a dimension reducer and a classifier can be set separately
	"""
	pass
