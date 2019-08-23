#!/usr/bin/env python3

import abc
# custom lib
import pylib.util


class ModelStructureAbstract(pylib.util.serializer.SerializerAbstract):
	"""
	abstract class of model structures that should be inherited by all model
	subclasses;
	"""
	@abc.abstractmethod
	def fit(self, X, Y):
		return self
	@abc.abstractmethod
	def predict(self, X, Y = None):
		pass
