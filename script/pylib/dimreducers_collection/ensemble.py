#!/usr/bin/env python3
###############################################################################
# this library implements custom ensemble dimension reduction methods
###############################################################################

import numpy
import sklearn.decomposition
import sklearn.model_selection
# custom lib
from . import base


@base.DimReducerCollection.register("lsdr_kpca_ensemble")
@base.DimReducerAbstract.serialize_init(as_name = "lsdr_kpca_ensemble",
	params = ["n_components", "n_estimators", "lsdr_n_classes", "lsdr_penalty",
		"lsdr_n_components", "lsdr_subsample_frac"])
class LSDR_KPCA_Ensemble(base.DimReducerAbstract):
	"""
	ensemble methods uses both HSIC-LSDR (as first stage) and KPCA (second
	stage):

	stage 1: HSIC-LSDR ensemble with subsampling
		train <n_estimators> independent LSDR, each with <lsdr_n_components>
		output dimensions with <lsdr_subsample_frac * total_train_samples>
		randomly drawn, training data (with replacement). all output transforms
		will be concatenated to a single output. this will result in a dataset
		with <n_estimators * lsdr_n_components> dimensions.
	stage 2: KPCA on LSDR ensemble output
		train a KPCA model with the output from stage 1 ensemble output. this
		further reduces the dimensions to final <n_components>.

	PARAMETERS
	----------
	n_components:
		number of final components output from stage 2;
	n_estimators:
		number of HSIC-LSDR used in stage 1 (default: 20);
	lsdr_n_classes:
		expected number of unique class labels to instruct HSIC-LSDR models;
	lsdr_penalty:
		penalty of HSIC-LSDR models (currently ineffective);
	lsdr_n_components:
		number of output components (default: 5)
	lsdr_subsample_frac:
		fraction of subsampling used in each train of HISC-LSDR in stage 1;
		(default: 0.6)
	"""
	def __init__(self, n_components, n_estimators = 20, lsdr_n_classes = 2,
		lsdr_penalty = 0.0, lsdr_n_components = 5, lsdr_subsample_frac = 0.6):
		self.n_components			= n_components
		self.n_estimators			= n_estimators
		self.lsdr_n_classes			= lsdr_n_classes
		self.lsdr_penalty			= lsdr_penalty
		self.lsdr_n_components		= lsdr_n_components
		self.lsdr_subsample_frac	= lsdr_subsample_frac
		self._lsdr = list()
		self._kpca = None
		return

	def _fit_lsdr_ensemble(self, X, Y):
		# train stage 1 lsdr ensemble
		cv = sklearn.model_selection.StratifiedShuffleSplit(random_state = None,
			n_splits = self.n_estimators,
			test_size = 1.0 - self.lsdr_subsample_frac)
		# clear old results
		self._lsdr.clear()
		for train, _ in cv.split(X, Y):
			# train independently <self.n_estimators> lsdr models
			lsdr = base.DimReducerCollection.query("lsdr")(
				n_components = self.lsdr_n_components,
				n_classes = self.lsdr_n_classes,
				penalty = self.lsdr_penalty)
			lsdr.fit(X[train], Y[train])
			self._lsdr.append(lsdr)
		return self

	def _transform_lsdr_ensemble(self, X):
		lsdr_trans = [lsdr.transform(X) for lsdr in self._lsdr]
		return numpy.hstack(lsdr_trans)

	def _fit_kpca(self, X, Y, *ka, **kw):
		self._kpca = base.DimReducerCollection.query("kpca")(
			n_components = self.n_components)
		lsdr_trans = self._transform_lsdr_ensemble(X)
		self._kpca.fit(lsdr_trans, Y, *ka, **kw)
		return self

	def fit(self, X, Y, *ka, use_default_gamma = True, **kw):
		self._fit_lsdr_ensemble(X, Y)
		self._fit_kpca(X, Y, *ka, use_default_gamma = use_default_gamma, **kw)
		return self

	def transform(self, X):
		lsdr_trans = self._transform_lsdr_ensemble(X)
		return self._kpca.transform(lsdr_trans)

	def serialize(self, *ka, **kw):
		ret = super(LSDR_KPCA_Ensemble, self).serialize(*ka, **kw)
		# add ensemble models
		lsdr_serialize = [lsdr.serialize() for lsdr in self._lsdr]
		kpca_serialize = self._kpca.serialize()
		ret["estimators"] = dict(lsdr = lsdr_serialize, kpca = kpca_serialize)
		return ret

	@classmethod
	def deserialze(cls, ds):
		# we don't need estimators info here
		stripped = {k: v for k, v in ds.items() if k != "estimators"}
		return super(LSDR_KPCA_Ensemble, cls).deserialze(stripped)
