#!/usr/bin/env python3

import sys
import os
import sklearn.metrics
from . import cv_split, dim_reduction, classifier, cv_result_plot


class CrossValidation(object):
	# result eval criteria for overall
	_OVERALL_KEYS_ = ["precision", "fscore", "accuracy"]
	# result eval criteria available in each class
	_CLASSES_KEYS_ = ["precision", "fscore"]

	def __init__(self, classifier, n_fold = 10,
		dim_reduc = None, reduce_dim_to = None):
		#
		super(CrossValidation, self).__init__()
		self.classifier = classifier
		self.n_fold = n_fold
		self.dim_reduc = dim_reduc
		self.reduce_dim_to = reduce_dim_to
		self.result = []


	def run_cv(self, X, Y):
		"""
		split dataset then run cross validation
		"""
		self.result.clear()
		# split data for cross validation
		escv = cv_split.EvenSplitCrossValidation(X, Y, self.n_fold)
		for train_data, test_data, train_label, test_label in escv:
			# dim reduction
			# self.dim_reduc returns a Plain object
			# which is a dummy dim reduction (does nothing)
			# just for avoiding the if ... else ... here
			dr = dim_reduction.get_dim_reduction_object(\
				model = self.dim_reduc, reduce_dim_to = self.reduce_dim_to)
			# dim reducetion on train data, then transform test data
			train_data = dr.fit_transform(train_data, train_label)
			test_data = dr.transform(test_data)
			# classifier
			cls = classifier.get_classifier_object(\
				self.classifier, data = train_data)
			# train classifier
			cls.fit(train_data, train_label)
			predict_label = cls.predict(test_data)
			# evaluate
			_res = self._evaluate(test_label, predict_label)
			# save results
			self.result.append(_res)
		return


	def _evaluate(self, y_true, y_pred):
		"""
		evaluate performance in each cross validation split
		"""
		acc_all = sklearn.metrics.accuracy_score(y_true, y_pred)
		prc_all = sklearn.metrics.precision_score(y_true, y_pred,
			average = "micro")
		prc_cls = sklearn.metrics.precision_score(y_true, y_pred,
			average = None)
		fs_all = sklearn.metrics.f1_score(y_true, y_pred,
			average = "micro")
		fs_cls = sklearn.metrics.f1_score(y_true, y_pred,
			average = None)
		#auc_all = sklearn.metrics.roc_auc_score(y_true, y_pred,
		#	average = "micro")
		#auc_cls = sklearn.metrics.roc_auc_score(y_true, y_pred,
		#	average = None)
		ret = {}
		ret["accuracy"] = dict(overall = acc_all)
		ret["precision"] = dict(overall = prc_all, classes = prc_cls)
		ret["fscore"] = dict(overall = fs_all, classes = fs_cls)
		#ret["auc"] = dict(overall = auc_all, classes = auc_cls)
		return ret


	def get_result(self):
		return self.result


	def savetxt(self, prefix, uniq_labels, label_encoder,
		*, delimiter = "\t"):
		# result structure: result[#fold]{key}{...}
		result = self.get_result()
		# save overall
		# each row is eval criteria
		# each col is a cv fold
		with open("%s.overall.txt" % prefix, "w") as fh:
			for key in self._OVERALL_KEYS_:
				vals = [i[key]["overall"] for i in result]
				vals_str = ["%f" % i for i in vals]
				print(delimiter.join([key] + vals_str),
					file = fh)
		# save eval criteria for each class
		# each row is a class
		# each col is a cv fold
		for key in self._CLASSES_KEYS_:
			with open("%s.%s.txt" % (prefix, key), "w") as fh:
				for label in uniq_labels:
					encoded, = label_encoder.transform([label])
					vals = [i[key]["classes"][encoded] for i in result]
					vals_str = ["%f" % i for i in vals]
					print(delimiter.join([label] + vals_str),
						file = fh)
		return


	def savefig_boxplot(self, *ka, **kw):
		cv_result_plot.boxplot(self, *ka, **kw)
		return
