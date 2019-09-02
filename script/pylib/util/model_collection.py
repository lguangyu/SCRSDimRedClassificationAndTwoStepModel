#!/usr/bin/env python3

import functools
# custom lib
from . import collection_registry
from . import serializer


class ModelCollection(collection_registry.CollectionRegistryBase):
	"""
	defines methods useful as a collection of models;
	"""
	@classmethod
	@functools.wraps(collection_registry.CollectionRegistryBase.register)
	def register(registry, *ka, **kw):
		"""
		on top of CollectionRegistryBase.register, this function also assumes
		the registered class is serializable (subclasses SerializerAbstract);
		"""
		def decorator(cls):
			if not issubclass(cls, serializer.SerializerAbstract):
				raise TypeError("elements of ModelCollection must be subclass "
					"of serializer.SerializerAbstract (serializable), not "
					"'%s'" % cls.__name__)
			return super(ModelCollection, registry).register(*ka, **kw)(cls)
		return decorator

	@classmethod
	def create(cls, key, *ka, **kw):
		"""
		get the factory from collection by key then create its instance;

		ARGUMENT
		key: the name of the model to query;
		*ka, **kw: other keyargs/kwargs passed to queried model constructor;
		"""
		return cls.query(key)(*ka, **kw)

	@classmethod
	def from_serialized(cls, serial):
		"""
		re-create a model using serialized data; similar to actual inverse of
		model.serialize();
		"""
		return cls.query(serial["model"]).deserialize(serial)
