#!/usr/bin/env python3

import numpy
import sklearn.metrics


class RBFKernelRoutinesMixin(object):
	"""
	useful functions used in RBF kernel calculations
	"""
	@staticmethod
	def rbf_gamma_by_median(X):
		"""
		return the gamma calculated with sigma as median of pairwise euclidean
		distances
		"""
		euc = sklearn.metrics.pairwise_distances(X, metric = "euclidean")
		sigma = numpy.median(euc)
		gamma = 0.5 / (sigma ** 2)
		return gamma
