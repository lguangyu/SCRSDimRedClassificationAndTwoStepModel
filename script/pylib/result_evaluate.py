#!/usr/bin/env python3

import abc
import numpy
import sklearn.metrics


class LabelPredictEvaluate(dict):
	def __init__(self, *ka, true_label, pred_label, **kw):
		super(LabelPredictEvaluate, self).__init__(*ka, **kw)
		self.evaluate(true_label, pred_label)
		return

	def evaluate(self, true_label, pred_label):
		self["average_accuracy"] = sklearn.metrics.accuracy_score(\
			true_label, pred_label)
		self["average_precision"] = sklearn.metrics.precision_score(\
			true_label, pred_label, average = "micro")
		self["class_precision"] = list(sklearn.metrics.precision_score(\
			true_label, pred_label, average = None))
		self["average_fscore"] = sklearn.metrics.f1_score(\
			true_label, pred_label, average = "micro")
		self["class_fscore"] = list(sklearn.metrics.f1_score(\
			true_label, pred_label, average = None))
		self["average_recall"] = sklearn.metrics.recall_score(\
			true_label, pred_label, average = "micro")
		self["class_recall"] = list(sklearn.metrics.recall_score(\
			true_label, pred_label, average = None))
		#self["average_auc"] = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		#self["class_auc"] = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		return


class ResultEvaluationBase(dict, abc.ABC):
	_MODE_KEYS_ = ["testing", "training"]

	def __init__(self, *ka, **kw):
		super(ResultEvaluationBase, self).__init__(*ka, **kw)
		return

	@abc.abstractmethod
	def evaluate(self, key, *ka, **kw):
		pass


class SingleLevelModelEvaluation(ResultEvaluationBase):
	def evaluate(self, key, true_label, pred_label):
		# check key
		if key not in self._MODE_KEYS_:
			raise ValueError("key must be one of the following: %s"\
				% repr(self._MODE_KEYS_))
		# evaluate
		self[key] = LabelPredictEvaluate(\
			true_label = true_label,\
			pred_label = pred_label)
		return


class TwoLevelModelEvaluation(ResultEvaluationBase):
	def evaluate(self, key,\
		lv1_true_label, lv1_pred_label,\
		lv2_true_label, lv2_pred_label):
		# check key
		if key not in self._MODE_KEYS_:
			raise ValueError("key must be one of the following: %s"\
				% repr(self._MODE_KEYS_))
		# evaluate
		ev = {}
		# level 1
		ev["level_1"] = LabelPredictEvaluate(\
			true_label = lv1_true_label,\
			pred_label = lv1_pred_label)
		# level 2
		ev_lv2 = {"per_lv1": [], "overall": None}
		for i in sorted(numpy.unique(lv1_true_label)):
			mask = (lv1_true_label == i)
			ev_lv2["per_lv1"].append(LabelPredictEvaluate(\
				true_label = lv2_true_label[mask],\
				pred_label = lv2_pred_label[mask]))
		ev_lv2["overall"] = LabelPredictEvaluate(\
			true_label = lv2_true_label,\
			pred_label = lv2_pred_label)
		ev["level_2"] = ev_lv2
		self[key] = ev
		return
