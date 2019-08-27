#!/usr/bin/env python3

import numpy


def as_unique(query_func, sequential):
	"""
	ensure query_func(sequential) call return results as an identical value;
	"""
	vals = set(map(query_func, sequential))
	if len(vals) != 1: # if unique, it should only contain one element
		raise ValueError("expected unique, got %d value(s): %s"\
			% (len(vals), str(vals)))
	return vals.pop()


class SingleLevelJSONCondenser(object):
	"""
	condense json output form 1-level model run output;
	"""
	def __init__(self, *ka, **kw):
		super(SingleLevelJSONCondenser, self).__init__(*ka, **kw)
		self.dataset		= None # dataset used by this model
		self.dimreducer		= None # dimensionality reduction method
		self.n_components	= None # number of dimensions after dr
		self.classifier		= None # classifier method
		self.metric			= None # extracted evaluation metric
		self.train_mean		= None # mean training evaluation by <metric>
		self.train_std		= None # std of training evaluation by <metric>
		self.test_mean		= None # mean testing evaluation by <metric>
		self.test_std		= None # std testing evaluation by <metric>
		return

	@staticmethod
	def _nd_query_func(round_res):
		"""
		helper function to parse n_components from each cross-validation round;
		since dimreducer's params field may be None;
		"""
		dr_par = round_res["dimreducer"]["params"]
		return None if dr_par is None else dr_par["n_components"]

	@staticmethod
	def _get_metric_stat(all_rounds_res, metric, testing = True):
		qry = lambda x:\
			x["evaluation"]["testing" if testing else "training"][metric]
		vals = list(map(qry, all_rounds_res))
		mean = numpy.mean(vals)
		std = numpy.std(vals)
		return mean, std

	@classmethod
	def parse(cls, res: dict, metric: str, *ka, **kw):
		if not isinstance(res, dict):
			raise TypeError("res must be dict, not '%s'" % type(res).__name__)
		if res["mode"] != "1-level":
			raise ValueError("result must run using '1-level' mode, not '%s'"\
				% res["mode"])
		# buiding basic information
		new = cls(*ka, **kw)
		new.dataset = res["dataset"]
		new.metric = metric
		# parsing these fields, assuming values are unique in all cv-rounds
		_rs = res["results"]
		new.dimreducer = as_unique(lambda x: x["dimreducer"]["model"], _rs)
		new.classifier = as_unique(lambda x: x["classifier"]["model"], _rs)
		new.n_components = as_unique(new._nd_query_func, _rs)
		# parsing evaluation result from cv-rounds
		new.train_mean, new.train_std = new._get_metric_stat(_rs, metric, False)
		new.test_mean, new.test_std = new._get_metric_stat(_rs, metric, True)
		return new
