#!/usr/bin/env python3

import collections
import numpy
from . import base_class
from . import level_based_model
from . import result_evaluate


class SingleLevelModel(level_based_model.LevelBasedModelBase):
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
	def level_props(self):
		return self._level_props
	@level_props.setter
	def level_props(self, value):
		if isinstance(value, self.LevelProps):
			self._level_props = value
			return
		elif isinstance(value, dict):
			self._level_props = self.LevelProps(**value)
			return
		raise ValueError("level_props must be LevelProps")

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
	def __init__(self, *ka, level_props, **kw):
		super(SingleLevelModel, self).__init__(*ka, **kw)
		self.level_props = level_props
		return

	def train(self, X, Y):
		# new result
		self.evaluation = self.ModelEvaluationResult()
		# dimension reduction first
		self.level_props.dim_reducer_obj.train(X, Y)
		# reduce dimension of X
		dr_X = self.level_props.dim_reducer_obj.transform(X)
		# then do classifier training, based on the transformed X
		self.level_props.classifier_obj.train(dr_X, Y)
		# do train evaluation
		pred_Y = self.predict(X) # FIXME: this is redundant calculation tho
		self.evaluation.evaluate("training", Y, pred_Y)
		return

	def predict(self, X):
		dr_X = self.level_props.dim_reducer_obj.transform(X)
		return self.level_props.classifier_obj.predict(dr_X)

	def test(self, test_X, test_Y):
		"""
		evalute the prediction of test dataset
		"""
		pred_Y = self.predict(test_X)
		self.evaluation.evaluate("testing", test_Y, pred_Y)
		return
