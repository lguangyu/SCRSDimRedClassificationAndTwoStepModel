#!/usr/bin/env python3

import collections
import numpy
from . import base_class
from . import classifier
from . import dim_reducer
from . import result_evaluate


class LevelProps(object):
	@property
	def dims_remain(self):
		return self._dims_remain
	@dims_remain.setter
	def dims_remain(self, value):
		if (value is None) or (value > 0):
			self._dims_remain = value
			return
		raise ValueError("dims after reduction must be positive")

	@property
	def classifier_obj(self):
		if not hasattr(self, "_classifier_obj"):
			self.new_classifier_obj()
		return self._classifier_obj

	@property
	def dim_reducer_obj(self):
		if not hasattr(self, "_dim_reducer_obj"):
			self.new_dim_reducer_obj()
		return self._dim_reducer_obj

	def __init__(self, *, dim_reducer, classifier, dims_remain = None):
		super(LevelProps, self).__init__()
		self.dim_reducer = dim_reducer
		self.dims_remain = dims_remain
		self.classifier = classifier
		return

	def new_classifier_obj(self, **kw):
		self._classifier_obj = classifier.create(self.classifier, **kw)
		return self._classifier_obj

	def new_dim_reducer_obj(self, **kw):
		self._dim_reducer_obj = dim_reducer.create(self.dim_reducer,
			n_components = self.dims_remain, **kw)
		return self._dim_reducer_obj


class SingleLevelModel(base_class.ABCModel):
	@property
	def level_props(self):
		return self._level_props
	@level_props.setter
	def level_props(self, value):
		if isinstance(value, LevelProps):
			self._level_props = value
			return
		elif isinstance(value, dict):
			self._level_props = LevelProps(**value)
			return
		raise ValueError("level_props must be LevelProps")

	@property
	def evaluation(self):
		return self._evaluation
	@evaluation.setter
	def evaluation(self, value):
		if isinstance(value, result_evaluate.SingleLevelModelEvaluation):
			self._evaluation = value
			return
		raise ValueError("evaluation must be SingleLevelModelEvaluation")

	############################################################################
	# methods
	def __init__(self, *ka, level_props, **kw):
		super(SingleLevelModel, self).__init__(*ka, **kw)
		self.level_props = level_props
		return

	def train(self, X, Y):
		# new result
		self.evaluation = result_evaluate.SingleLevelModelEvaluation()
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


class TwoLevelModel(base_class.ABCModel):
	@property
	def level1_props(self):
		return self._level1_props
	@level1_props.setter
	def level1_props(self, value):
		if isinstance(value, LevelProps):
			self._level1_props = value
			return
		elif isinstance(value, dict):
			self._level1_props = LevelProps(**value)
			return
		raise ValueError("level1_props must be LevelProps")

	@property
	def level2_props(self):
		return self._level2_props
	@level2_props.setter
	def level2_props(self, value):
		if isinstance(value, LevelProps):
			self._level2_props = value
			return
		elif isinstance(value, dict):
			self._level2_props = LevelProps(**value)
			return
		raise ValueError("level2_props must be LevelProps")

	@property
	def evaluation(self):
		return self._evaluation
	@evaluation.setter
	def evaluation(self, value):
		if isinstance(value, result_evaluate.TwoLevelModelEvaluation):
			self._evaluation = value
			return
		raise ValueError("evaluation must be TwoLevelModelEvaluation")

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
		self.evaluation = result_evaluate.TwoLevelModelEvaluation()
		# level 1
		self.level1_props.dim_reducer_obj.train(X, level1_Y)
		dr_X = self.level1_props.dim_reducer_obj.transform(X)
		self.level1_props.classifier_obj.train(dr_X, level1_Y)
		# predict level 1 label
		pred_lv1_Y = self.level1_props.classifier_obj.predict(dr_X)
		level2_splits = self._mask_level2_by_level1(\
			(level1_Y if self.indep_level2 else pred_lv1_Y), X, level2_Y)
		# reset the level2 model list
		self.level2_props.splits = {}
		for i, l2x, l2y in level2_splits:
			# create new objects
			self.level2_props.new_dim_reducer_obj()
			self.level2_props.new_classifier_obj()
			# train
			self.level2_props.dim_reducer_obj.train(l2x, l2y)
			dr_X = self.level2_props.dim_reducer_obj.transform(l2x)
			self.level2_props.classifier_obj.train(dr_X, l2y)
			# record
			self.level2_props.splits[i] = {\
				"dim_reducer_obj": self.level2_props.dim_reducer_obj,\
				"classifier_obj": self.level2_props.classifier_obj}
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
			l2x = l2x.reshape(1, -1)
			# use the trained model for given level 1 label
			level2_split = self.level2_props.splits[l1y]
			dr_X = level2_split["dim_reducer_obj"].transform(l2x)
			assert dr_X.shape[0] == 1, dr_X.shape
			_pred, = level2_split["classifier_obj"].predict(dr_X)
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
