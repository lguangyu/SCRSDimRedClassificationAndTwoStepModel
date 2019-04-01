#!/usr/bin/env python3

import sklearn.metrics


class LabelPredictEvaluate(object):
	def __init__(self, true_label, pred_label, *ka, **kw):
		super(LabelPredictEvaluate, self).__init__(*ka, **kw)
		self.evaluate(true_label, pred_label)
		return

	def evaluate(self, true_label, pred_label):
		overall_accuracy = sklearn.metrics.accuracy_score(true_label, pred_label)
		overall_precision = sklearn.metrics.precision_score(true_label, pred_label,\
			average = "micro")
		class_precision = sklearn.metrics.precision_score(true_label, pred_label,\
			average = None)
		overall_fscore = sklearn.metrics.f1_score(true_label, pred_label,\
			average = "micro")
		class_fscore = sklearn.metrics.f1_score(true_label, pred_label,\
			average = None)
		#
		self.overall_accuracy = overall_accuracy
		self.overall_precision = overall_precision
		self.class_precision = class_precision
		self.overall_fscore = overall_fscore
		self.class_fscore = class_fscore
		return

	def dump_txt(self, file):
		print("\t".join(["accuracy:", "%f" % self.overall_accuracy]), file = file)
		print("\t".join(["precision:", "%f" % self.overall_precision]), file = file)
		print("\t".join(["per-class-precision:"]\
			+ ["%f" % i for i in self.class_precision]), file = file)
		print("\t".join(["fscore", "%f" % self.overall_fscore]), file = file)
		print("\t".join(["per-class-fscore:"]\
			+ ["%f" % i for i in self.class_fscore]), file = file)
		return
