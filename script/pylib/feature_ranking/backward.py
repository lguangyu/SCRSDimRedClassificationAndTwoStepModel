#!/usr/bin/env python3

import abc
import numpy
import sklearn.metrics
import sklearn.preprocessing
# custom lib
from . import base
import pylib.util


class FeatureRankBackwardBase(base.FeatureRankAbstract):
	@abc.abstractmethod
	def calc_relevence(self, X, Y = None) -> float:
		"""
		the method to calculate the relevence of X to Y; a larger relevence
		means the data in X preserves better information for learning tasks
		towards label Y; must return a single float value:
		"""
		pass

	def rank_features(self, X, Y = None) -> numpy.ndarray:
		if X.ndim != 2:
			raise ValueError("X must be 2-dim ndarray")

		ascend_rank = list()
		remain_feat = set(range(X.shape[1]))

		while len(remain_feat) >= 2:
			# calculate the score of each feature
			remain_feat_list = list(remain_feat)
			relev_list = list()
			for f in remain_feat_list:
				feat_mask = [i for i in remain_feat_list if i != f]
				X_masked = X[:, feat_mask]
				relev = self.calc_relevence(X_masked, Y = Y)
				relev_list.append(relev)

			# find the feature that resulted in maximum relevence when removed
			relev_order = numpy.argsort(relev_list)
			f = remain_feat_list[relev_order[-1]] # last one is the largest
			ascend_rank.append(f)
			# remove f from remain_feat to continue the outer loop
			remain_feat.remove(f)
		# add the last feature
		ascend_rank.append(remain_feat.pop())
		if remain_feat:
			raise RuntimeError("something went wrong, 'remain_feat' not empty")

		# flip the order due to backward selection
		ret = numpy.asarray(ascend_rank[::-1])
		return ret


@base.FeatureRankCollection.register("hsic", "bahsic")
class HSIC(FeatureRankBackwardBase,
		pylib.util.kernel_routines.RBFKernelRoutinesMixin):
	def calc_relevence(self, X, Y) -> float:
		n_samples = len(X)

		gamma = self.rbf_gamma_by_median(X)
		KH = self.centering(
			sklearn.metrics.pairwise.rbf_kernel(X, gamma = gamma),
			copy = False,
		)

		onehot_y = sklearn.preprocessing.OneHotEncoder().fit_transform(
			Y.reshape(-1, 1)
		).toarray()
		LH = self.centering(numpy.dot(onehot_y, onehot_y.T), copy = False)

		hsic = numpy.trace(numpy.dot(KH, LH)) / (n_samples ** 2)
		return hsic


