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

	@staticmethod
	def get_centering_matrix(degree):
		"""
		return a centering matrix of degree n, i.e.
			H = I_n - (1/n) (*) 1_n
		* I_n: identity matrix of degree n
		* 1_n: square full-of-1 matrix of degree n
		"""
		H = numpy.eye(degree) - numpy.full((degree, degree), 1 / degree)
		return H

	@staticmethod
	def centering(matrix, copy = True):
		"""
		return a matrix both row- and column-wise centered from input;
		this is equivalent to 
			matrix (*) H
		* H: centering matrix of degree same as #features in X
		while we calculate it using mean-subtraction, for better performance on
		large matrices than dot multiplication
		"""
		if matrix.ndim != 2:
			raise ValueError("input matrix must be 2-d, got %d-d" % matrix.ndim)
		ret = matrix.copy() if copy else matrix
		ret -= ret.mean(axis = 0, keepdims = True)
		ret -= ret.mean(axis = 1, keepdims = True)
		return ret
