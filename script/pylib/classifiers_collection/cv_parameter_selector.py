#!/usr/bin/env python3

import itertools
import numpy
import sklearn.model_selection
# custom lib
from pylib.evaluator import ClassifEvaluator


class CVClassifParamsDict(dict):
	def expand(self):
		"""
		CVClassifParamsDict is in format:
		{"A": [1, 2], "B": [3]}

		expand into a list of dicts:
		[{"A": 1, "B": 3}, {"A": 2, "B": 3}]

		each dict in in the expanded list can be used in model.set_params() as:
		for params in cv_params.expand():
			model.set_params(**params)
		"""
		keys = sorted(self.keys())
		pars_mesh = itertools.product(*map(lambda k: self[k], keys))
		return [dict(zip(keys, pars)) for pars in pars_mesh]


class _CVClassifParamSelectMixin(object):
	"""
	cross validation routines, must be used as a mixin with ClassifierAbstract;
	.fit() method calls parent class' relative to do the actual training in each
	split, but the parameter selection is done in CV's .fit() method;
	the actual invoked .fit() will be dependent to the inheritance [as mro()];
	"""
	def _cv_param_mean(self, X, Y, param, *ka, cv_props, eval_metric, **kw):
		"""
		run over a complete round of k-fold cv through k splits given a set of
		parameters, return the mean evaluation (e.g. accuracy or precision) from
		all splits' testing sets;
		"""
		self.set_params(**param)
		# cv here, use stratified k fold to ensure even split of each class
		evals = list()
		cv = sklearn.model_selection.StratifiedKFold(**cv_props)
		for train, test in cv.split(X, Y):
			# and this is why this classed must be used as a mixin;
			# check mro() for actual .fit() invoked here;
			super(_CVClassifParamSelectMixin, self)\
				.fit(X[train], Y[train], *ka, **kw)
			# note this prevents call subclass.predict() per bare .predict()
			# invocation; this ensures matched .fit() and .predict() from the
			# same model class
			pred = super(_CVClassifParamSelectMixin, self).predict(X[test])
			evals.append(ClassifEvaluator.evaluate(Y[test], pred)[eval_metric])
		return numpy.mean(evals)

	def fit(self, X, Y, *ka, cv_params: dict, cv_props: dict = None,
			eval_metric: str = "average_accuracy", **kw) -> "self":
		"""
		train model with cross-validation; all combinations of parameters
		provided by <cv_params> dict will be evaluated via cross validation
		specified by <cv_props>; the final model will be set and re-trained
		per best parameter combination attempted;

		ARGUMENT
		X, Y: follow .fit() convention;
		cv_params: a dict of parameters (and their list of values) to be tried
			and selected by cross validation;
		cv_props: properties dict passed to StratifiedKFold cross validator;
			default: {"n_splits": 10, "shuffle": True}
		eval_metric: the metric to evaluate model fit; see ClassifEvaluator for
			a complete list of acceptable keys; default: "average_accuracy";
		*ka, **kw: other keyargs/kwargs passed to parent class' .fit() method;
		"""
		if not isinstance(cv_params, dict):
			raise TypeError("cv_params must be dict, not '%s'"\
				% type(cv_params).__name__)
		if not cv_params:
			raise ValueError("cv_params cannot be empty dict")
		if cv_props is None:
			cv_props = dict(n_splits = 10, shuffle = True)
		# coerce cv_params to CVClassifParamsDict class if is plain dict
		#cv_params = cv_params if isinstance(cv_params, CVClassifParamsDict)\
		#	else CVClassifParamsDict(cv_params)
		# then safe to expand
		#paras_expanded = cv_params.expand()
		paras_expanded = CVClassifParamsDict.expand(cv_params) # well this is ok
		evals = [self._cv_param_mean(X, Y, i, *ka, cv_props = cv_props,
			eval_metric = eval_metric, **kw) for i in paras_expanded]
		# pick the best parameters
		best_id = numpy.argmax(evals)
		best_para = paras_expanded[best_id]
		# retrain with best parameters
		self.set_params(**best_para)
		return super(_CVClassifParamSelectMixin, self).fit(X, Y, *ka, **kw)
