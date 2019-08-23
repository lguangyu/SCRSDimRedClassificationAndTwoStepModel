#!/usr/bin/env python3

import abc
# custom lib
from . import serializer


class ModelSerializerBase(serializer.SerializerAbstract):
	"""
	base serializer interface for models (classifiers, dim-reducers, etc.);
	_serialize_as_ is a string class-member showing the name of the serialized
	model, and _serialize_params_ is a list class-member defining list of
	parameters should be serialized;

	METHODS

	.serialize(): return a dict have at least two keys: 
		'model': the name of the serialized model object;
		'params': a dict of {param_name : param_value}; if _serialize_params_
			is None, params is None;

	.deserialze(): return a re-constructed object;
	"""
	@staticmethod
	def serialize_init(as_name: str, params: set or list = None) -> "decorator":
		"""
		decorator factory to setup the information for serialization; correct
		settings of these parameters will be required (and checked) by either
		.serialize() or .deserialze() methods;
		parent class's _serialize_params_ list will be copied to subclass;
		"""
		# check parameters
		if not isinstance(as_name, str):
			raise TypeError("as_name must be str not '%s'"\
				% type(as_name).__name__)
		if not (isinstance(params, set) or isinstance(params, list) or
			(params is None)):
			raise TypeError("params must be None, list or set, not '%s'"\
				% type(params).__name__)
		# actual decorator
		def decorator(cls):
			if not issubclass(cls, ModelSerializerBase):
				# must used on class, NOT instance of that class
				raise TypeError("this decorator must be used on subclass of "
					"ModelSerializerBase, not '%s'" % type(cls).__name__)
			cls._serialize_as_ = as_name
			# copy parent _serialize_params_ field
			_pars = set()
			for b in cls.mro(): # OPT: cls.mro()[1:] to exclude cls
				_pars.update(getattr(b, "_serialize_params_", None) or set())
			if (not _pars) and (params is None):
				cls._serialize_params_ = None
			else:
				_pars.update(set() if params is None else params)
				cls._serialize_params_ = _pars
			return cls
		return decorator

	@classmethod
	def _check_serialize_settings(cls):
		"""
		check if _serialize_as_ and _serialize_params_ has been correctly set;
		"""
		if getattr(cls, "_serialize_as_", None) is None:
			raise cls.SerializerError("serialization requires %s be decorated "
				"by ModelSerializerBase.serialize_init() (_serialize_as_ not "
				"set)" % cls.__name__)
		return

	def serialize(self, *ka, **kw):
		self._check_serialize_settings()
		ret = dict(model = type(self)._serialize_as_)
		# subclass must superset parent class._serialize_params_
		_par_keys = getattr(type(self), "_serialize_params_", None)
		# so we dont need to recurse upwards
		ret["params"] = None if _par_keys is None else\
			{k: self.get_params()[k] for k in _par_keys}
		return ret

	@classmethod
	def deserialze(cls, ds):
		cls._check_serialize_settings()
		# restrict to be exact match, subclass not allowed
		if cls._serialize_as_ != ds["model"]:
			raise TypeError("cannot deserialze model '%s' as '%s'"\
				% (ds["model"], cls._serialize_as_))
		new = cls(**(dict() if ds["params"] is None else ds["params"]))
		return new
