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
			true_label, pred_label, average = "macro")
		self["class_precision"] = list(sklearn.metrics.precision_score(\
			true_label, pred_label, average = None))
		self["average_fscore"] = sklearn.metrics.f1_score(\
			true_label, pred_label, average = "macro")
		self["class_fscore"] = list(sklearn.metrics.f1_score(\
			true_label, pred_label, average = None))
		self["average_recall"] = sklearn.metrics.recall_score(\
			true_label, pred_label, average = "macro")
		self["class_recall"] = list(sklearn.metrics.recall_score(\
			true_label, pred_label, average = None))
		#self["average_auc"] = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		#self["class_auc"] = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		return
