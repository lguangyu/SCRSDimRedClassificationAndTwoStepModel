#!/usr/bin/env python3

import functools
import numpy
import sklearn.preprocessing
# custom lib
import pylib.util


class cached_property(property):
	def __init__(self, fini, **kw): # **kw is omitted anyway...
		self.prop_name = "_cp_" + fini.__name__
		########################################################################
		# wrapped fget and fdel; fset is omitted
		# <inst> is the instance of object this property resides in
		@functools.wraps(fini)
		def fget(inst):
			if not hasattr(inst, self.prop_name):
				setattr(inst, self.prop_name, fini(inst))
			return getattr(inst, self.prop_name)
		def fdel(inst):
			if hasattr(inst, self.prop_name):
				delattr(inst, self.prop_name)
			return
		# 
		return super(cached_property, self).__init__(fget, None, fdel, **kw)


class DatasetBase(object):
	"""
	dataset used by models;
	"""
	############################################################################
	# maybe overridden to read from other files
	_raw_data_file_		= "./data/ALL.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/ALL.label.txt"

	############################################################################
	# routines of preprocessing raw dataset into specific dataset
	@cached_property
	def raw_data(self):
		return numpy.loadtxt(self._raw_data_file_, float, delimiter = "\t")

	@cached_property
	def raw_label(self):
		return numpy.loadtxt(self._raw_label_file_, object, delimiter = "\t")

	@property
	def raw_phase_label(self):
		return self.raw_label[:, 0]

	@property
	def raw_strain_label(self):
		return self.raw_label[:, 1]

	############################################################################
	# preprocess pipeline routines
	def pp_scale(self, data, *ka, **kw):
		return sklearn.preprocessing.scale(data, *ka, **kw)

	def pp_filter(self, cond, *data, axis = 0, **kw):
		return tuple([numpy.compress(cond, d, axis = axis, **kw) for d in data])

	def pp_encode_label(self, label):
		encoder = sklearn.preprocessing.LabelEncoder()
		return encoder, encoder.fit_transform(label)

	def __init__(self, *ka, **kw):
		super(DatasetBase, self).__init__(*ka, **kw)
		return


class SingleLabelDataset(DatasetBase):
	"""
	dataset that has only one label category;

	SingleLabelDataset should have below properties:
	data:			main data matrix
	label:			labels encoded as numerical values
	text_label:		labels in text format
	label_encoder:	LabelEncoder instance fitted by <text_label>
	"""
	pass


@pylib.util.collection_registry.CollectionRegistryBase.init(DatasetBase)
class DatasetCollection(pylib.util.collection_registry.CollectionRegistryBase):
	"""
	collection registry of all datasets classes; all elements should subclass
	DatasetBase to be recogized as valid;
	"""
	@classmethod
	def get_dataset(cls, key, *ka, **kw):
		"""
		create a new instance of queried dataset class;

		ARGUMENT
		key: the name of the dataset to query;
		*ka, **kw: other keyargs/kwargs passed to queried dataset constructor;
		"""
		return cls.query(key)(*ka, **kw)
		

################################################################################
# datasets
class PhaseDatasetBase(SingleLabelDataset):
	############################################################################
	# subclass must override this to extract a different phase
	_extract_phase_ = None

	def __init__(self, *ka, **kw):
		super(PhaseDatasetBase, self).__init__(*ka, **kw)
		_label, _data = self.pp_filter(
			(self.raw_phase_label == self._extract_phase_), # condition
			self.raw_strain_label, self.raw_data) # filtered list
		self.data = self.pp_scale(_data)
		self.text_label = _label
		self.label_encoder, self.label = self.pp_encode_label(self.text_label)
		return


@DatasetCollection.register("exponential")
class ExponentialPhaseDataset(PhaseDatasetBase):
	_extract_phase_ = "EXPONENTIAL"


@DatasetCollection.register("platform-1", "stationary-1")
class Platform1PhaseDataset(PhaseDatasetBase):
	_extract_phase_ = "PLATFORM1"


@DatasetCollection.register("platform-2", "stationary-2")
class Platform2PhaseDataset(PhaseDatasetBase):
	_extract_phase_ = "PLATFORM2"


@DatasetCollection.register("strain-only")
class StrainLabelDataset(SingleLabelDataset):
	def __init__(self, *ka, **kw):
		super(StrainLabelDataset, self).__init__(*ka, **kw)
		self.data = self.pp_scale(self.raw_data)
		self.text_label = self.raw_strain_label
		self.label_encoder, self.label = self.pp_encode_label(self.text_label)
		return


@DatasetCollection.register("phase-only")
class PhaseLabelDataset(SingleLabelDataset):
	def __init__(self, *ka, **kw):
		super(PhaseLabelDataset, self).__init__(*ka, **kw)
		self.data = self.pp_scale(self.raw_data)
		self.text_label = self.raw_phase_label
		self.label_encoder, self.label = self.pp_encode_label(self.text_label)
		return


@DatasetCollection.register("phase-and-strain", "duo-label")
class DuoLabelDataset(DatasetBase):
	"""
	datasets use both phase and strain labels;

	DuoLabelDataset should have below properties:
	data:					main data matrix (scaled)
	phase_label:			phase labels encoded as numerical values
	phase_text_label:		phase labels in text format
	phase_label_encoder:	LabelEncoder instance fitted by <phase text_label>
	strain_label:			strain labels encoded as numerical values
	strain_text_label:		strain labels in text format
	strain_label_encoder:	LabelEncoder instance fitted by <strain text_label>
	"""
	def __init__(self, *ka, **kw):
		super(DuoLabelDataset, self).__init__(*ka, **kw)
		self.data = self.pp_scale(self.raw_data)
		self.phase_text_label = self.raw_phase_label
		self.phase_label_encoder, self.phase_label\
			= self.pp_encode_label(self.phase_text_label)
		self.strain_text_label = self.raw_strain_label
		self.strain_label_encoder, self.strain_label\
			= self.pp_encode_label(self.strain_text_label)
		return
