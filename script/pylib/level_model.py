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
	def train_evaluation(self):
		return self._train_evaluation

	@property
	def test_evaluation(self):
		return self._test_evaluation

	############################################################################
	# methods
	def __init__(self, *ka, level_props, **kw):
		super(SingleLevelModel, self).__init__(*ka, **kw)
		self.level_props = level_props
		return

	def train(self, X, Y):
		# dimension reduction first
		self.level_props.dim_reducer_obj.train(X, Y)
		# reduce dimension of X
		dr_X = self.level_props.dim_reducer_obj.transform(X)
		# then do classifier training, based on the transformed X
		self.level_props.classifier_obj.train(dr_X, Y)
		# do train evaluation
		self._train_evaluation = self.evaluate(X, Y)
		return

	def predict(self, X):
		dr_X = self.level_props.dim_reducer_obj.transform(X)
		return self.level_props.classifier_obj.predict(dr_X)

	def evaluate(self, X, true_Y):
		"""
		evaluate the prediction of X's label, w.r.t. ground truth Y
		equivalent to call predict(X) followed by evaluation
		"""
		pred_Y = self.predict(X)
		return result_evaluate.LabelPredictEvaluate(true_Y, pred_Y)

	def test(self, test_X, test_Y):
		"""
		evalute the prediction of test dataset
		"""
		self._test_evaluation = self.evaluate(test_X, test_Y)
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
	def train_evaluation(self):
		return self._train_evaluation

	@property
	def test_evaluation(self):
		return self._test_evaluation

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
		# level 1
		self.level1_props.dim_reducer_obj.train(X, level1_Y)
		dr_X = self.level1_props.dim_reducer_obj.transform(X)
		self.level1_props.classifier_obj.train(dr_X, level1_Y)
		# decide the splits of level 2 training
		if self.indep_level2:
			l1y = level1_Y
		else:
			l1y = self.level1_props.classifier_obj.predict(dr_X)
		level2_splits = self._mask_level2_by_level1(l1y, X, level2_Y)
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
		self._train_evaluation = self.evaluate(X, level2_Y)
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
		return numpy.asarray(pred_level2_labels, dtype = int)

	def evaluate(self, X, true_Y):
		"""
		evaluate the prediction of X's label, w.r.t. ground truth Y
		equivalent to call predict(X) followed by evaluation
		"""
		pred_Y = self.predict(X)
		return result_evaluate.LabelPredictEvaluate(true_Y, pred_Y)

	def test(self, test_X, test_Y):
		"""
		evalute the prediction of test dataset
		"""
		self._test_evaluation = self.evaluate(test_X, test_Y)
		return
