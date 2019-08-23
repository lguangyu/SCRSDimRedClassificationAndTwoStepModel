#!/usr/bin/env python3

import abc
import json


class SerializerError(RuntimeError):
	pass

class SerializerAbstract(abc.ABC):
	"""
	interface to change formats to/from standard built-in data structures, list
	and dict;
	"""
	@abc.abstractmethod
	def serialize(self):
		"""
		interface to serialze self to built-in data structures; inverse to
		cls.deserialze()
		"""
		pass

	@classmethod
	@abc.abstractmethod
	def deserialze(cls, ds):
		"""
		interface to create a new instance of cls from serialized data; inverse
		to .serialize()
		"""
		pass


class SerializerJSONEncoder(json.JSONEncoder):
	"""
	add json-compatible interface to handling subclass of SerializerAbstract;
	this transform in one-directional, from custom data class instance to built-
	in data structures;

	SYNOPSIS

	class Foo(SerializerAbstract):
		def serialize(...):
			...
		def invert_serialze(...):
			...

	# a foo instance now fulfills the SerializerAbstract protocol, it can be
	# handled by SerializerJSONEncoder in json.dump()
	foo = Foo()
	json.dumps(foo, cls = SerializerJSONEncoder)
	"""
	def default(self, o, *ka, **kw):
		if isinstance(o, SerializerAbstract):
			return o.serialize()
		return super(SerializerJSONEncoder, self).default(o, *ka, **kw)
