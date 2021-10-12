#!/usr/bin/env python3

import numpy
import sys
# custom lib
from . import base
# interface to ism_sdr package
from . import _ism_sdr


@base.DimReducerCollection.register("ism_sdr")
@base.DimReducerAbstract.serialize_init(as_name = "ism_sdr",
	params = ["n_components", "penalty"])
class ISM_SDR(base.DimReducerAbstract):
	"""
	wrapper class of iterative spectrum method - supervised dimension reduction
	(ISM-SDR) in linear mode;

	PARAMETERS
	n_components: dimensionality after reduction;
	n_classes (DEPRECATED): number of classes in labels input to the supervised
		dr training, ignored; depcrecated due to package 'lsdr' update, kept for
		forward competibility;
	penalty: regularizer (currently ineffective);
	"""
	def __init__(self, n_components, n_classes = 2, penalty = 0.0):
		self.n_components = n_components
		self.n_classes = n_classes
		self.penalty = penalty
		return

	def fit(self, X, Y):
		self._sdr = _ism_sdr.sdr.sdr(X, Y, self.n_components)
		self._sdr.train()
		return self

	def transform(self, X):
		W = self._sdr.get_projection_matrix()
		return numpy.dot(X, W)


#@base.DimReducerCollection.register("lsdr_reg", "regularized_hsic_lsdr")
#@base.DimReducerAbstract.serialize_init(as_name = "lsdr_reg")
#class Regularized_HSIC_LSDR(HSIC_LSDR):
#	pass
