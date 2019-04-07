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
