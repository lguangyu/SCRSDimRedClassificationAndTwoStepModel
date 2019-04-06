#!/usr/bin/env python3

import collections
from . import base_class
from . import classifier as classifier_m
from . import dim_reducer as dim_reducer_m


class LevelPropsBase(object):
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

	############################################################################
	# methods
	def __init__(self, *ka, dim_reducer, classifier, dims_remain = None, **kw):
		super(LevelPropsBase, self).__init__(*ka, **kw)
		self.dim_reducer = dim_reducer
		self.dims_remain = dims_remain
		self.classifier = classifier
		return

	def _get_classifier_obj(self, **kw):
		return classifier_m.create(self.classifier, **kw)


	def _get_dim_reducer_obj(self, **kw):
		return dim_reducer_m.create(self.dim_reducer,\
			n_components = self.dims_remain, **kw)

	def duplicate(self, astype = None):
		"""
		clone self settings to a new LevelPropsBase instance
		"""
		if astype is None:
			astype = type(self)
		if not issubclass(astype, LevelPropsBase):
			raise TypeError("astype must be subclass of LevelPropsBase")
		# copy
		new = astype(\
			dim_reducer = self.dim_reducer,\
			classifier = self.classifier,\
			dims_remain = self.dims_remain)
		return new


class LevelPropsGeneral(LevelPropsBase):
	"""
	LevelProps subclass only manages one classifier/dim reducer object
	"""
	@property
	def classifier_obj(self):
		"""
		get the instance of classifier
		"""
		if not hasattr(self, "_classifier_obj"):
			self._classifier_obj = self._get_classifier_obj()
		return self._classifier_obj

	@property
	def dim_reducer_obj(self):
		"""
		get the instance of classifier
		"""
		if not hasattr(self, "_dim_reducer_obj"):
			self._dim_reducer_obj = self._get_dim_reducer_obj()
		return self._dim_reducer_obj


class LevelPropsWithSplit(LevelPropsBase, collections.defaultdict):
	"""
	LevelProps subclass only manages several LevelPropsGeneral instances per
	query label (a.k.a. split)
	each split can be get with __getitem__ method
	"""
	def __init__(self, *ka, **kw):
		print("2", super(LevelPropsWithSplit, self).__init__)
		super(LevelPropsWithSplit, self).__init__(\
			lambda : self.duplicate(astype = LevelPropsGeneral), *ka, **kw)
			# default split is copy settings from self
		return


class LevelBasedModelBase(base_class.ABCModel):
	"""
	model base class which can be separated into levels (ok if #levels = 1)
	in each level, a dimension reducer and a classifier can be set separately
	"""
	LevelProps = LevelPropsGeneral
	LevelPropsSplit = LevelPropsWithSplit
