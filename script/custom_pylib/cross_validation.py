#!/usr/bin/env python3

from . import cv_split, dim_reduction, classifier


class CrossValidation(object):
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
		self.result.clear()
		# split data for cross validation
		escv = cv_split.EvenSplitCrossValidation(X, Y, self.n_fold)
		for train_data, test_data, train_label, test_label in escv:
			# dim reduction
			# self.dim_reduc returns a Plain object
			# which is a dummy dim reduction (does nothing)
			# just for avoiding the if ... else ... here
			dr = dim_reduction.get_dim_reduction_object(
				self.dim_reduc, self.reduce_dim_to)
			# dim reducetion on train data, then transform test data
			train_data = dr.fit_transform(train_data)
			test_data = dr.transform(test_data)
			# classifier
			cls = classifier.get_classifier_object(self.classifier)
			# train classifier
			cls.fit(train_data, train_label)
			predict_label = cls.predict(test_data)
			# evaluate
			_res = self._evaluate(predict_label, test_label)
			# save results
			self.result.append(_res)
		return


	def _evaluate(self, predict_label, actual_label):
		pass


	def get_result(self):
		return self.result
