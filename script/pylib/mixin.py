#!/usr/bin/env python3

import abc
import collections.abc


class RegularizerMixin(object):
	"""
	dimension reducer/classifier with regularizer as parameter
	"""
	@property
	def regularizer(self):
		return self._regularizer
	@regularizer.setter
	def regularizer(self, value):
		self._regularizer = value
		return

	def __init__(self, *ka, regularizer = 0.0, **kw):
		super(RegularizerMixin, self).__init__(*ka, **kw)
		self.regularizer = regularizer
		return


class RegularizerSelectorMixin(object):
	"""
	select tuning parameter for dimension reducer/classifier with regularizer
	"""
	@property
	def regularizer_list(self):
		return self._regularizer_list
	@regularizer_list.setter
	def regularizer_list(self, value):
		if (value is None) or isinstance(value, collections.abc.Iterable):
			self._regularizer_list = value
			return
		raise ValueError("regularizer_list must be iterable")

	def __init__(self, *ka, regularizer_list = None, **kw):
		super(RegularizerSelectorMixin, self).__init__(*ka, **kw)
		self.regularizer_list = regularizer_list
		return
