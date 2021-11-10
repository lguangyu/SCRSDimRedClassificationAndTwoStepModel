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
	_raw_data_file_		= "./data/oxford_28.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/oxford_28.labels.txt"

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


class SingleLabelDatasetBase(DatasetBase):
	"""
	dataset has only one label category, either phase or strain;

	SingleLabelDatasetBase should have below properties:
	data:			main data matrix
	label:			labels encoded as numerical values
	text_label:		labels in text format
	label_encoder:	LabelEncoder instance fitted by <text_label>
	"""
	pass


class DuoLabelDatasetBase(DatasetBase):
	"""
	dataset use both phase and strain labels;

	DuoLabelDatasetBase should have below properties:
	data:					main data matrix (scaled)
	phase_label:			phase labels encoded as numerical values
	phase_text_label:		phase labels in text format
	phase_label_encoder:	LabelEncoder instance fitted by <phase text_label>
	strain_label:			strain labels encoded as numerical values
	strain_text_label:		strain labels in text format
	strain_label_encoder:	LabelEncoder instance fitted by <strain text_label>
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

		ARGUMENTS
		key: the name of the dataset to query;
		*ka, **kw: other keyargs/kwargs passed to queried dataset constructor;
		"""
		return cls.query(key)(*ka, **kw)

	@classmethod
	def check_query_subclass(cls, key, exp_cls) -> bool:
		"""
		return True if the dataset queried by <key> is of <exp_cls> type, False
		otherwise;

		ARGUMENTS
		key: the name of the dataset to query;
		exp_cls: expected dataset class type, must be subclass of DatasetBase;
		"""
		if not (isinstance(exp_cls, type) and issubclass(exp_cls, DatasetBase)):
			raise TypeError("exp_cls must be a subclass of DatasetBase, "
				"not '%s'" % type(exp_cls).__name__)
		return issubclass(cls.query(key), exp_cls)


################################################################################
# datasets
class PhaseDatasetBase(SingleLabelDatasetBase):
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


################################################################################
# oxford datasets
@DatasetCollection.register("oxford-exponential")
class OxfordExponentialPhaseDataset(PhaseDatasetBase):
	_extract_phase_ = "EXPONENTIAL"


@DatasetCollection.register("oxford-platform-1", "oxford-stationary-1")
class OxfordPlatform1PhaseDataset(PhaseDatasetBase):
	_extract_phase_ = "PLATFORM1"


@DatasetCollection.register("oxford-platform-2", "oxford-stationary-2")
class OxfordPlatform2PhaseDataset(PhaseDatasetBase):
	_extract_phase_ = "PLATFORM2"


@DatasetCollection.register("oxford-strain-only")
class OxfordStrainLabelDataset(SingleLabelDatasetBase):
	def __init__(self, *ka, **kw):
		super(OxfordStrainLabelDataset, self).__init__(*ka, **kw)
		self.data = self.pp_scale(self.raw_data)
		self.text_label = self.raw_strain_label
		self.label_encoder, self.label = self.pp_encode_label(self.text_label)
		return


@DatasetCollection.register("oxford-phase-only")
class OxfordPhaseLabelDataset(SingleLabelDatasetBase):
	def __init__(self, *ka, **kw):
		super(OxfordPhaseLabelDataset, self).__init__(*ka, **kw)
		no_st2_mask = (self.raw_phase_label != "PLATFORM2")
		_label, _data = self.pp_filter(
			no_st2_mask, # condition
			self.raw_phase_label, self.raw_data) # filtered list
		self.data = self.pp_scale(_data)
		self.text_label = _label
		self.label_encoder, self.label = self.pp_encode_label(self.text_label)
		return


@DatasetCollection.register("oxford-phase-and-strain", "oxford-duo-label")
class OxfordDuoLabelDataset(DuoLabelDatasetBase):
	def __init__(self, *ka, **kw):
		super(OxfordDuoLabelDataset, self).__init__(*ka, **kw)
		no_st2_mask = (self.raw_phase_label != "PLATFORM2")
		_phase_label, _strain_label, _data = self.pp_filter(
			no_st2_mask, # condition
			# filtered list
			self.raw_phase_label, self.raw_strain_label, self.raw_data)
		self.data = self.pp_scale(_data)
		self.phase_text_label = _phase_label
		self.phase_label_encoder, self.phase_label\
			= self.pp_encode_label(self.phase_text_label)
		self.strain_text_label = _strain_label
		self.strain_label_encoder, self.strain_label\
			= self.pp_encode_label(self.strain_text_label)
		return


################################################################################
# zijian datasets
@DatasetCollection.register("zijian-exponential")
class ZijianExponentialPhaseDataset(PhaseDatasetBase):
	_raw_data_file_		= "./data/zijian_40.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/zijian_40.labels.txt"
	_extract_phase_ = "Exponential"


@DatasetCollection.register("zijian-stationary-1")
class ZijianStationary1PhaseDataset(PhaseDatasetBase):
	_raw_data_file_		= "./data/zijian_40.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/zijian_40.labels.txt"
	_extract_phase_ = "Stationary1"


@DatasetCollection.register("zijian-stationary-2")
class ZijianStationary2PhaseDataset(PhaseDatasetBase):
	_raw_data_file_		= "./data/zijian_40.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/zijian_40.labels.txt"
	_extract_phase_ = "Stationary2"


@DatasetCollection.register("zijian-stationary-3")
class ZijianStationary3PhaseDataset(PhaseDatasetBase):
	_raw_data_file_		= "./data/zijian_40.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/zijian_40.labels.txt"
	_extract_phase_ = "Stationary3"


@DatasetCollection.register("zijian-phase-only")
class ZijianPhaseLabelDataset(SingleLabelDatasetBase):
	_raw_data_file_		= "./data/zijian_40.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/zijian_40.labels.txt"

	def __init__(self, *ka, **kw):
		super(ZijianPhaseLabelDataset, self).__init__(*ka, **kw)
		self.data = self.pp_scale(self.raw_data)
		self.text_label = self.raw_phase_label
		self.label_encoder, self.label = self.pp_encode_label(self.text_label)
		return


@DatasetCollection.register("zijian-phase-and-strain", "zijian-duo-label")
class ZijianDuoLabelDataset(DuoLabelDatasetBase):
	_raw_data_file_		= "./data/zijian_40.normalized_l2.data.tsv"
	_raw_label_file_	= "./data/zijian_40.labels.txt"

	def __init__(self, *ka, **kw):
		super(ZijianDuoLabelDataset, self).__init__(*ka, **kw)
		self.data = self.pp_scale(self.raw_data)
		self.phase_text_label = self.raw_phase_label
		self.phase_label_encoder, self.phase_label\
			= self.pp_encode_label(self.phase_text_label)
		self.strain_text_label = self.raw_strain_label
		self.strain_label_encoder, self.strain_label\
			= self.pp_encode_label(self.strain_text_label)
		return
