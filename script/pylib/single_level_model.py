#!/usr/bin/env python3

import collections
import numpy
import sklearn.metrics
import sklearn.model_selection
from . import base_class
from . import level_based_model
from . import mixin
from . import result_evaluate


class SingleLevelModel(level_based_model.LevelBasedModelBase,\
	level_based_model.LevelPropsMixin, mixin.RegularizerSelectorMixin):
	############################################################################
	# member class
	class ModelEvaluationResult(base_class.ResultEvaluationBase):
		_MODE_KEYS_ = ["testing", "training"]

		def evaluate(self, key, true_label, pred_label):
			# check key
			if key not in self._MODE_KEYS_:
				raise ValueError("key must be one of the following: %s"\
					% repr(self._MODE_KEYS_))
			# evaluate
			self[key] = result_evaluate.LabelPredictEvaluate(\
				true_label = true_label,\
				pred_label = pred_label)
			return

	############################################################################
	# variables
	@property
	def evaluation(self):
		return self._evaluation
	@evaluation.setter
	def evaluation(self, value):
		if isinstance(value, self.ModelEvaluationResult):
			self._evaluation = value
			return
		raise ValueError("evaluation must be "\
			+ "SingleLevelModel.ModelEvaluationResult")

	############################################################################
	# methods
	def __init__(self, *ka, **kw):
		super(SingleLevelModel, self).__init__(*ka, **kw)
		return

	def train(self, X, Y):
		# new result
		self.evaluation = self.ModelEvaluationResult()
		if self.regularizer_list is not None:
			self._select_regularizer(X, Y)
		else:
			self._train(X, Y)
		# do train evaluation
		pred_Y = self.predict(X) # FIXME: this is redundant calculation tho
		self.evaluation.evaluate("training", Y, pred_Y)
		return

	def _train(self, X, Y):
		# dimension reduction first
		self.dim_reducer_obj.train(X, Y)
		# reduce dimension of X
		dr_X = self.dim_reducer_obj.transform(X)
		# then do classifier training, based on the transformed X
		self.classifier_obj.train(dr_X, Y)
		return

	def _select_regularizer(self, X, Y):
		reg_ev = []
		for reg in self.regularizer_list:
			print("regularizer:", reg)
			# only do with dim reducer/classifier with regularizer argument
			if isinstance(self.dim_reducer_obj, mixin.RegularizerMixin):
				self.new_dim_reducer_obj(regularizer = reg)
			if isinstance(self.classifier_obj, mixin.RegularizerMixin):
				self.new_classifier_obj(regularizer = reg)
			# cross validation
			cv_ev = []
			cv_splitter = sklearn.model_selection.StratifiedKFold(\
				n_splits = 10, shuffle = True)
			for train_indices, test_indices in cv_splitter.split(X, Y):
				# split
				train_X = X[train_indices]
				train_Y = Y[train_indices]
				test_X = X[test_indices]
				test_Y = Y[test_indices]
				self._train(train_X, train_Y)
				pred_Y = self.predict(test_X)
				# evaluation each round
				accuracy = sklearn.metrics.accuracy_score(test_Y, pred_Y)
				cv_ev.append(accuracy)
				print(accuracy)
			reg_ev.append(numpy.mean(cv_ev))
			print("average:", numpy.mean(cv_ev))
		# find the best
		_imax = numpy.argmax(reg_ev)
		_best_reg = self.regularizer_list[_imax]
		print("best:", _imax, ",", _best_reg)
		# retrain with best regularizer
		if isinstance(self.dim_reducer_obj, mixin.RegularizerMixin):
			self.new_dim_reducer_obj(regularizer = _best_reg)
		if isinstance(self.classifier_obj, mixin.RegularizerMixin):
			self.new_classifier_obj(regularizer = _best_reg)
		self._train(X, Y) # use all X and Y
		return

	def predict(self, X):
		dr_X = self.dim_reducer_obj.transform(X)
		return self.classifier_obj.predict(dr_X)

	def test(self, test_X, test_Y):
		"""
		evalute the prediction of test dataset
		"""
		pred_Y = self.predict(test_X)
		self.evaluation.evaluate("testing", test_Y, pred_Y)
		return
