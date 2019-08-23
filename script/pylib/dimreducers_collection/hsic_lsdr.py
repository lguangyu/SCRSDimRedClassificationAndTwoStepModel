#!/usr/bin/env python3

import numpy
import sys
# custom lib
from . import base
# interface to HSIC package
sys.path.append("./src")
import LSDR as lsdr_lib


@base.DimReducerCollection.register("lsdr", "hsic_lsdr")
@base.DimReducerAbstract.serialize_init(as_name = "lsdr",
	params = ["n_components", "n_classes", "penalty"])
class HSIC_LSDR(base.DimReducerAbstract):
	"""
	wrapper class of HISC linear supervised dimension reduction (LSDR);

	PARAMETERS
	n_components: dimensionality after reduction;
	n_classes: number of classes in labels input to the supervised training;
	penalty: regularizer (currently ineffective);
	"""
	def __init__(self, n_components, n_classes = 2, penalty = 0.0):
		self.n_components = n_components
		self.n_classes = n_classes
		self.penalty = penalty
		return

	def _make_db(self, X, Y):
		# num of classes in labels (Y)
		self.set_params(n_clusters = len(numpy.unique(Y)))
		# this is used by LSDR library routines
		ret = dict(X = X, Y = Y,
			num_of_clusters = self.n_classes,
			q = self.n_components,
			#λ_ratio = 0.0,
			λ = self.penalty,
			center_and_scale = False # we do it manually outside
		)
		self._db = ret
		return ret

	def fit(self, X, Y):
		db = self._make_db(X, Y)
		self._sdr = lsdr_lib.LSDR(db)
		self._sdr.train()
		return self

	def transform(self, X):
		W = self._sdr.get_projection_matrix()
		return numpy.dot(X, W)


@base.DimReducerCollection.register("lsdr_reg", "regularized_hsic_lsdr")
@base.DimReducerAbstract.serialize_init(as_name = "lsdr_reg")
class Regularized_HSIC_LSDR(HSIC_LSDR):
	pass
