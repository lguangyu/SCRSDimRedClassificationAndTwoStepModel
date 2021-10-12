#!/usr/bin/env python3

import functools
import sklearn.metrics
# custom lib
import pylib.util


class ClassifEvaluator(dict, pylib.util.serializer.SerializerAbstract):
	"""
	evaluate classifier performance by calculating various scores from testing
	label and predicted label; evaluations are done using class method
	.evaluate(); the evaluator is an enhanced dict also supports using keys
	directly as attributes;

	SYNOPSIS
	>>> foo = ClassifEvaluator.evaluate(true_label, pred_label)
	>>> print(foo["average_accuracy"]) # use as dict, OR equivalently,
	>>> print(foo.average_accuracy) # use as attribute
	"""
	@classmethod
	def evaluate(cls, true_label, pred_label):
		new = cls()
		new["true_label"] = [int(i) for i in true_label]
		new["pred_label"] = [int(i) for i in pred_label]
		new["average_accuracy"] = sklearn.metrics.accuracy_score(\
			true_label, pred_label)
		new["average_precision"] = sklearn.metrics.precision_score(\
			true_label, pred_label, average = "macro")
		new["class_precision"] = list(sklearn.metrics.precision_score(\
			true_label, pred_label, average = None))
		new["average_fscore"] = sklearn.metrics.f1_score(\
			true_label, pred_label, average = "macro")
		new["class_fscore"] = list(sklearn.metrics.f1_score(\
			true_label, pred_label, average = None))
		new["average_recall"] = sklearn.metrics.recall_score(\
			true_label, pred_label, average = "macro")
		new["class_recall"] = list(sklearn.metrics.recall_score(\
			true_label, pred_label, average = None))
		#self["average_auc"] = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		#self["class_auc"] = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		return new

	def __getattr__(self, attr):
		try:
			return getattr(self, attr)
		except AttributeError as e:
			if attr in self:
				return self[attr] # search from dict
			raise

	def serialize(self):
		return self

	@classmethod
	def deserialize(cls, ds):
		return cls(ds)


class ModelEvaluationResultsMixin(object):
	"""
	mixin routines support organizing testing and training evaluations;
	"""
	def __init__(self, *ka, **kw):
		super(ModelEvaluationResultsMixin, self).__init__(*ka, **kw)
		self._ev = dict(training = None, testing = None)
		return

	def _check_arg_which(allow_all = False):
		"""
		decorator factory to check the value of 'which' used intensively by this
		class' methods; 'which' must be the second positional argument of all
		decorated functions;
		by default, which is only allowed to be any key presents in self._ev;

		ARGUMENT
		allow_all: if true, also allow 'all' as an option;
		"""
		def decorator(func):
			@functools.wraps(func, assigned = ("__defaults__", ))
			def wrapper(self, which, *ka, **kw):
				opt = list(self._ev.keys()) + (["all"] if allow_all else [])
				if which not in opt:
					raise KeyError("which must be one of: %s, not '%s'"\
						% (str(opt), which))
				return func(self, which, *ka, **kw)
			# apply positional args default to wrapper
			#wrapper.__defaults__ = func.__defaults__
			return wrapper
		return decorator

	@_check_arg_which(allow_all = True)
	def reset_eval_results(self, which = "all"):
		if which == "all":
			for key in self._ev.keys():
				self._ev[key] = None
		else:
			self._ev[which] = None
		return

	@_check_arg_which(allow_all = True)
	def get_eval_results(self, which = "all"):
		if which == "all":
			return self._ev.copy()
		else:
			return self._ev[which]

	@_check_arg_which(allow_all = False)
	def set_eval_results(self, which, result):
		if not isinstance(result, ClassifEvaluator):
			raise TypeError("input evaluation result must be ClassifEvaluator, "
				"not '%s'" % type(result).__name__)
		self._ev[which] = result
		return
