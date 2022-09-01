#!/usr/bin/env python3

import functools
from skfeature.function.similarity_based.fisher_score import fisher_score
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.function.similarity_based.trace_ratio import trace_ratio
# custom lib
from . import base


@base.FeatureRankCollection.register("fisher_score")
class FisherScore(base.FeatureRankAbstract):
	def rank_features(self, X, Y):
		return fisher_score(X, Y, mode = "index")


@base.FeatureRankCollection.register("laplacian_score", "lap_score")
class LaplacianScore(base.FeatureRankAbstract):
	def rank_features(self, X, Y):
		return lap_score(X, Y, mode = "index")


@base.FeatureRankCollection.register("trace_ratio")
class TraceRatio(base.FeatureRankAbstract):
	def rank_features(self, X, Y):
		return trace_ratio(X, Y, mode = "index")
