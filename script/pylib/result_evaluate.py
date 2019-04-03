#!/usr/bin/env python3

import sklearn.metrics


class LabelPredictEvaluate(object):
	def __init__(self, true_label, pred_label, *ka, **kw):
		super(LabelPredictEvaluate, self).__init__(*ka, **kw)
		self.evaluate(true_label, pred_label)
		return

	def evaluate(self, true_label, pred_label):
		self.overall_accuracy = sklearn.metrics.accuracy_score(\
			true_label, pred_label)
		self.overall_precision = sklearn.metrics.precision_score(\
			true_label, pred_label, average = "micro")
		self.class_precision = sklearn.metrics.precision_score(\
			true_label, pred_label, average = None)
		self.overall_fscore = sklearn.metrics.f1_score(\
			true_label, pred_label, average = "micro")
		self.class_fscore = sklearn.metrics.f1_score(\
			true_label, pred_label, average = None)
		self.overall_recall = sklearn.metrics.recall_score(\
			true_label, pred_label, average = "micro")
		self.class_recall = sklearn.metrics.recall_score(\
			true_label, pred_label, average = None)
		self.overall_auc = sklearn.metrics.roc_auc_score(\
			true_label, pred_label, average = "micro")
		#self.class_auc = sklearn.metrics.roc_auc_score(\
		#	true_label, pred_label, average = None)
		return

	def __repr__(self):
		s = []
		_vars = vars(self)
		for k in sorted(_vars.keys()):
			if "overall_" in k:
				s.append("\t".join([k + ":", "%f" % _vars[k]]))
			elif "class_" in k:
				s.append("\t".join([k + ":"] + ["%f" % i for i in _vars[k]]))
			else:
				raise ValueError("unknown key: %s" % k)
		return "\n".join(s)
