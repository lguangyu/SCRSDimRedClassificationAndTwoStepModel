#!/usr/bin/env python3


class CollectionRegistryMetaclass(type):
	"""
	metaclass of CollectionRegistryBase; take care of essential class attributes
	_reg_dict and _reg_type upon class creation;
	"""
	def __new__(metaclass, name, bases, attrs):
		metaclass._check_attr(attrs, "_reg_dict", dict)
		metaclass._check_attr(attrs, "_key_alias_dict", dict)
		metaclass._check_attr(attrs, "_reg_type", type, factory = lambda : object)
		return super(CollectionRegistryMetaclass, metaclass).__new__(
			metaclass, name, bases, attrs)

	@staticmethod
	def _check_attr(attrs_dict, attr_key, exp_type, *, factory = None):
		"""
		check if attribute <attr_key> exists in attrs_dict; if not, first try
		create from factory() if <factory> is set, otherwise from exp_type();
		else if exists, check if is instance of <exp_type>, raise TypeError if
		not;
		"""
		if attr_key not in attrs_dict:
			val = factory() if factory is not None else exp_type()
			attrs_dict.update({attr_key: val})
		elif not isinstance(attrs_dict[attr_key], exp_type):
			raise TypeError("%s must be %s, not '%s'" % (attr_key,
				exp_type.__name__, type(attrs_dict[attr_key]).__name__))
		return


class CollectionRegistryBase(object, metaclass = CollectionRegistryMetaclass):
	"""
	registry table for managing collection of models; this base class must be
	subclassed to use;
	"""
	# these class-attributes are essential
	# subclass can manually define them in class definition, or use .init()
	# (as decorator)
	_reg_dict		= dict()
	_key_alias_dict	= dict()
	_reg_type		= object

	@classmethod
	def _check_is_subclassed(cls):
		"""
		prevent direct use of this base class
		"""
		if cls is CollectionRegistryBase:
			raise TypeError("CollectionRegistryBase must be subclassed to use")
		return

	@classmethod
	def get_registered_keys(cls):
		"""
		number of registered in registry
		"""
		cls._check_is_subclassed()
		return sorted(cls._reg_dict.keys())

	@classmethod
	def register(registry, primary_key, *alias_keys):
		"""
		decorator factory to register a model class to this collection
		"""
		registry._check_is_subclassed()
		def decorator(cls):
			if not issubclass(cls, registry._reg_type):
				raise TypeError("must register with a subclass of '%s', not "
					"'%s'" % (registry._reg_type.__name__, cls.__name__))
			# add key->cls to registry
			for k in (primary_key, *alias_keys):
				if k in registry._reg_dict:
					raise ValueError("key '%s' already registered" % k)
				registry._reg_dict[k] = cls
			# add primary->alias
			registry._key_alias_dict[primary_key] = alias_keys
			return cls
		return decorator

	@classmethod
	def query(cls, key):
		cls._check_is_subclassed()
		if key not in cls._reg_dict:
			raise KeyError("key '%s' not found in registry %s"\
				% (key, cls.__name__))
		return cls._reg_dict[key]

	@classmethod
	def repr_reg_keys(cls, show_alias = "discriminant") -> str:
		"""
		represent all registered keys in a string;

		ARGUMENTS
		show_alias:
			the strategy to show alias; if 'plain', all alias(es) will be listed
			the same as primary keys; if 'discriminant', alias(es) will be shown
			in parenthese after their primary key; if 'no', only primary keys
			will be shown (default: 'discriminant');
		"""
		if show_alias == "no":
			ret = (", ").join(sorted(cls._key_alias_dict.keys()))
		elif show_alias == "discriminant":
			key_alias_list = list()
			for k in sorted(cls._key_alias_dict.keys()):
				s = k
				if cls._key_alias_dict[k]:
					s += " [aka %s]"\
						% (", ").join(sorted(cls._key_alias_dict[k]))
				key_alias_list.append(s)
			ret = ("; ").join(key_alias_list)
		elif show_alias == "plain":
			ret = (", ").join(cls.get_registered_keys())
		else:
			raise ValueError("show_alias must be one of: 'no', 'discriminant' "
				"or 'plain', not '%s'" % show_alias)
		return ret

	@staticmethod
	def init(member_type):
		"""
		decorator factory used with CollectionRegistryBase to set the restricted
		type of elements in registry;
		"""
		if not isinstance(member_type, type):
			raise TypeError("member_type must be class, not '%s'"\
				% type(member_type).__name__)
		# actual decorator
		def decorator(registry):
			registry._check_is_subclassed()
			if not issubclass(registry, CollectionRegistryBase):
				raise TypeError("decorated class must be class/subclass of "
					"CollectionRegistryBase, not '%s'" % registry.__name__)
			registry._reg_type = member_type
			return registry
		return decorator
