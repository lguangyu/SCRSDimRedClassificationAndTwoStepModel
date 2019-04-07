#!/usr/bin/env python3

import abc
import collections
#import constructing
from . import base_class
from . import classifier as classifier_m
from . import dim_reducer as dim_reducer_m


class LevelPropsBase(abc.ABC):
	"""
	manager class of dim_reducer/classifier used in each level
	"""
	@abc.abstractmethod
	def clone_props_to(self, target: "LevelPropsBase"):
		"""
		clone self settings to the target level-props object
		these props are subclass-dependent
		"""
		pass


	def clone(self, astype: "LevelPropsBase" = None):
		"""
		create a new level-props object and clone self settings to it
		when astype is set, cast to that target type if compatible; otherwise
		use type(self)

		NOTE:
		compatibility should be checked explicitly by derived classes, in method
		clone_props_to()
		"""
		if astype is None:
			astype = type(self)
		if not issubclass(astype, LevelPropsBase):
			raise ValueError("astype must be LevelPropsBase")
		# using astype.__new__() bypasses astype.__init__()
		# since we will pass required arguments (by __init__()) later, not now
		new = astype.__new__(astype)
		self.clone_props_to(target = new)
		return new


class LevelPropsGeneralMixin(LevelPropsBase):
	"""
	general props manages one dim_reducer and one classifier
	these arguments are general information needed to initialize a level-based
	model for its dim_reducer and classifier
	"""
	############################################################################
	# variables
	@property
	def dims_remain(self):
		"""
		number of remaining dimensions after dimension reduction
		"""
		return self._dims_remain
	@dims_remain.setter
	def dims_remain(self, value):
		if (value is None) or (value > 0):
			self._dims_remain = value
			return
		raise ValueError("dims after reduction must be positive")

	@property
	def classifier(self):
		"""
		name of the classifier
		"""
		return self._classifier
	@classifier.setter
	def classifier(self, value):
		self._classifier = value
		return
	
	@property
	def dim_reducer(self):
		"""
		name of the dimension reducer
		"""
		return self._dim_reducer
	@dim_reducer.setter
	def dim_reducer(self, value):
		self._dim_reducer = value
		return

	@property
	def classifier_obj(self):
		"""
		instance of classifier object
		on first time query, create with 'classifier'
		"""
		if not hasattr(self, "_classifier_obj"):
			self.new_classifier_obj()
		return self._classifier_obj

	@property
	def dim_reducer_obj(self):
		"""
		instance of dimension reducer
		on first time query, create with 'dim_reducer'
		"""
		if not hasattr(self, "_dim_reducer_obj"):
			self.new_dim_reducer_obj()
		return self._dim_reducer_obj

	############################################################################
	# methods
	def __init__(self, *ka, dim_reducer, classifier, dims_remain = None, **kw):
		super(LevelPropsGeneralMixin, self).__init__(*ka, **kw)
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
		"""
		create a new classifier object, and replace the current one
		"""
		self._classifier_obj = self._get_classifier_obj(**kw)
		return self._classifier_obj

	def new_dim_reducer_obj(self, **kw):
		"""
		create a new dim_reducer object, and replace the current one
		"""
		self._dim_reducer_obj = self._get_dim_reducer_obj(**kw)
		return self._dim_reducer_obj

	def clone_props_to(self, target):
		if not isinstance(target, LevelPropsGeneralMixin):
			raise TypeError("target must be compatible with "\
				+ "LevelPropsRegularizerSelectionMixin")
		super(LevelPropsGeneralMixin, self).clone_props_to(target)
		target.dim_reducer = self.dim_reducer
		target.dims_remain = self.dims_remain
		target.classifier = self.classifier
		return


class LevelPropsRegularizerSelectionMixin(LevelPropsBase):
	"""
	select tuning parameter for dimension reducer/classifier with regularizer
	"""
	@property
	def regularizer_list(self):
		"""
		list of regularizer values to explore
		"""
		return self._regularizer_list
	@regularizer_list.setter
	def regularizer_list(self, value):
		if (value is None) or isinstance(value, collections.abc.Iterable):
			self._regularizer_list = value
			return
		raise ValueError("regularizer_list must be iterable")

	def __init__(self, *ka, regularizer_list = None, **kw):
		super(LevelPropsRegularizerSelectionMixin, self).__init__(*ka, **kw)
		self.regularizer_list = regularizer_list
		return

	def clone_props_to(self, target):
		if not isinstance(target, LevelPropsRegularizerSelectionMixin):
			raise TypeError("target must be compatible with "\
				+ "LevelPropsRegularizerSelectionMixin")
		super(LevelPropsRegularizerSelectionMixin, self).clone_props_to(target)
		target.regularizer_list = self.regularizer_list
		return


class LevelBasedModelBase(base_class.ABCModel):
	"""
	model base class which can be separated into levels (ok if #levels = 1)
	in each level, a dimension reducer and a classifier can be set separately

	currently a dummy class reserved for future use
	"""
	pass
