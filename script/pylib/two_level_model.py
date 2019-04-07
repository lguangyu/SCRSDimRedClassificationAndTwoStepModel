#!/usr/bin/env python3

import collections
import numpy
from . import base_class
from . import level_based_model
from . import single_level_model
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

	class Level2ModelSplit(level_based_model.LevelPropsRegularizerSelectionMixin,\
		level_based_model.LevelPropsGeneralMixin,\
		collections.defaultdict):
		"""
		LevelPropsBase subclass manages several LevelPropsGeneralMixin instances
		per query label (a.k.a. split)
		each split is a LevelPropsGeneralMixin instance, which can be accessed
		with __getitem__ method
		"""
		def __init__(self, *ka, **kw):
			super(TwoLevelModel.Level2ModelSplit, self).__init__(\
				lambda : self.clone(astype = single_level_model.SingleLevelModel),\
				*ka, **kw)
			# above lambda is passed to defaultdict.__init__ as default_factory
			# default split is copy settings from self
			return

	############################################################################
	# variables
	@property
	def level_1(self):
		return self._level1_model

	@property
	def level_2(self):
		return self._level2_model

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
		# level 1 model is a single-level model
		self._level1_model = single_level_model.SingleLevelModel(**level1_props)
		# level 2 model is a combination of single-level models
		self._level2_model = self.Level2ModelSplit(**level2_props)
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
		# level 1, and predict level 1 labels
		self.level_1.train(X, level1_Y)
		pred_lv1_Y = self.level_1.predict(X)
		# reset the level2 model list
		self.level_2.clear()
		# split for level 2
		for i, l2x, l2y in self._mask_level2_by_level1(\
			(level1_Y if self.indep_level2 else pred_lv1_Y), X, level2_Y):
			# above use ground truth level1_Y if use independent training manner;
			# else use predicted label pred_lv1_Y
			#
			# train each split
			# direct index [i] should be safe, as level2_props is defaultdict
			self.level_2[i].train(l2x, l2y)
		# predict level 2 label
		pred_lv1_Y, pred_lv2_Y = self.predict(X)
		self.evaluation.evaluate("training",\
			level1_Y, pred_lv1_Y,\
			level2_Y, pred_lv2_Y)
		return

	def predict(self, X):
		# level 1
		pred_level1_labels = self.level_1.predict(X)
		# level 2
		assert len(pred_level1_labels) == len(X), len(pred_level1_labels)
		pred_level2_labels = []
		for l1y, l2x in zip(pred_level1_labels, X):
			# since pred_level1_labels is mixed, have to do below one-by-one
			l2x = l2x.reshape(1, -1) # ensure 2-d array
			_pred, = self.level_2[l1y].predict(l2x)
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
