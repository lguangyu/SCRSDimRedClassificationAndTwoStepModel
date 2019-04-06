#!/usr/bin/env python3

import collections
import numpy
from . import base_class
from . import level_based_model
from . import result_evaluate


class TwoLevelModel(level_based_model.LevelBasedModelBase):
	############################################################################
	# member class
	class ModelEvaluationResult(base_class.ResultEvaluationBase):
		_MODE_KEYS_ = ["training", "testing"]

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
			ev["level_1"] = result_evaluate.LabelPredictEvaluate(\
				true_label = lv1_true_label,\
				pred_label = lv1_pred_label)
			# level 2
			ev_lv2 = {"per_lv1": [], "overall": None}
			for i in sorted(numpy.unique(lv1_true_label)):
				mask = (lv1_true_label == i)
				ev_lv2["per_lv1"].append(result_evaluate.LabelPredictEvaluate(\
					true_label = lv2_true_label[mask],\
					pred_label = lv2_pred_label[mask]))
			ev_lv2["overall"] = result_evaluate.LabelPredictEvaluate(\
				true_label = lv2_true_label,\
				pred_label = lv2_pred_label)
			ev["level_2"] = ev_lv2
			self[key] = ev
			return

	############################################################################
	# variables
	@property
	def level1_props(self):
		return self._level1_props
	@level1_props.setter
	def level1_props(self, value):
		if isinstance(value, self.LevelProps):
			self._level1_props = value
			return
		elif isinstance(value, dict):
			self.level1_props = self.LevelProps(**value)
			return
		raise ValueError("level1_props must be LevelProps or kwargs dict")

	@property
	def level2_props(self):
		return self._level2_props
	@level2_props.setter
	def level2_props(self, value):
		if isinstance(value, self.LevelPropsSplit):
			self._level2_props = value
			return
		elif isinstance(value, dict):
			self.level2_props = self.LevelPropsSplit(**value)
			return
		raise ValueError("level2_props must be LevelPropsSplit or kwargs dict")

	@property
	def evaluation(self):
		return self._evaluation
	@evaluation.setter
	def evaluation(self, value):
		if isinstance(value, self.ModelEvaluationResult):
			self._evaluation = value
			return
		raise ValueError("evaluation must be "\
			+ "TwoLevelModel.ModelEvaluationResult")

	############################################################################
	# methods
	def __init__(self, *ka, level1_props, level2_props,
		indep_level2 = False, **kw):
		"""
		indep_level2: if True, level2 model is trained independent to level1,
			directly from the ground truth (level1_Y labels); otherwise, trained
			from the labels predicted from level1 model;
			the independent training prevents the error propagation from level1
			prediction; the dependent training however is more closer to the
			application;
		"""
		super(TwoLevelModel, self).__init__(*ka, **kw)
		self.indep_level2 = indep_level2
		self.level1_props = level1_props
		self.level2_props = level2_props
		return

	def _mask_level2_by_level1(self, level1_Y, level2_X, level2_Y):
		uniq_labels = numpy.sort(numpy.unique(level1_Y))
		# below lambda select the corresponding records in level2_X and level2_Y
		# for each unique labels in level1_Y
		for i in uniq_labels:
			mask = (level1_Y == i)
			yield i, level2_X[mask], level2_Y[mask]

	def train(self, X, level1_Y, level2_Y):
		# new result
		self.evaluation = self.ModelEvaluationResult()
		# level 1
		self.level1_props.dim_reducer_obj.train(X, level1_Y)
		dr_X = self.level1_props.dim_reducer_obj.transform(X)
		self.level1_props.classifier_obj.train(dr_X, level1_Y)
		# predict level 1 label
		pred_lv1_Y = self.level1_props.classifier_obj.predict(dr_X)
		level2_splits = self._mask_level2_by_level1(\
			(level1_Y if self.indep_level2 else pred_lv1_Y), X, level2_Y)
		# reset the level2 model list
		self.level2_props.clear()
		for i, l2x, l2y in level2_splits:
			# train
			# direct index [i] should be safe, as level2_props is defaultdict
			self.level2_props[i].dim_reducer_obj.train(l2x, l2y)
			dr_X = self.level2_props[i].dim_reducer_obj.transform(l2x)
			self.level2_props[i].classifier_obj.train(dr_X, l2y)
		# predict level 2 label
		pred_lv1_Y, pred_lv2_Y = self.predict(X)
		self.evaluation.evaluate("training",\
			level1_Y, pred_lv1_Y,\
			level2_Y, pred_lv2_Y)
		return

	def predict(self, X):
		# level 1
		dr_X = self.level1_props.dim_reducer_obj.transform(X)
		pred_level1_labels = self.level1_props.classifier_obj.predict(dr_X)
		# level 2
		assert len(pred_level1_labels) == len(X), len(pred_level1_labels)
		pred_level2_labels = []
		for l1y, l2x in zip(pred_level1_labels, X):
			# since pred_level1_labels is mixed, have to do below one-by-one
			l2x = l2x.reshape(1, -1) # ensure 2-d array
			# use the trained model per given level 1 label
			dr_X = self.level2_props[l1y].dim_reducer_obj.transform(l2x)
			assert dr_X.shape[0] == 1, dr_X.shape
			_pred, = self.level2_props[l1y].classifier_obj.predict(dr_X)
			pred_level2_labels.append(_pred)
		pred_level2_labels = numpy.asarray(pred_level2_labels, dtype = int)
		return pred_level1_labels, pred_level2_labels

	def test(self, test_X, test_lv1_Y, test_lv2_Y):
		"""
		evalute the prediction of test dataset
		"""
		pred_lv1_Y, pred_lv2_Y = self.predict(test_X)
		self.evaluation.evaluate("testing",\
			test_lv1_Y, pred_lv1_Y,\
			test_lv2_Y, pred_lv2_Y)
		return
