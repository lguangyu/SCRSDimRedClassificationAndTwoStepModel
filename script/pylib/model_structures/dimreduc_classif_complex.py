#!/usr/bin/env python3

import functools
import gc
# custom lib
from . import base
import pylib.classifiers_collection
import pylib.dimreducers_collection
import pylib.evaluator
import pylib.util


class DimredClassifComplex(pylib.evaluator.ModelEvaluationResultsMixin,
		base.ModelStructureAbstract):
	"""
	a model complex that combines a dimension reducer and a classifier to
	perform a two-step pipeline training/prediction; input data will be reduced
	dimensionality first then perform classification training based on labels;
	"""
	def __init__(self, dimreducer_props: dict, classifier_props: dict,
			*ka, **kw):
		super(DimredClassifComplex, self).__init__(*ka, **kw)
		self.dimreducer_props = dimreducer_props
		self.classifier_props = classifier_props
		self.__dr = None # the dimreducer object
		self.__cf = None # the classifier object
		self.reset_eval_results("all")
		return

	def get_dimreducer(self, *, force_create = False):
		"""
		get the dimension reducer object;
		if not initialized, or force_create = True, create and return a new one;
		NOTE: create a new object in any case will reset all evaluation results;
		"""
		if ((self.__dr is None) or force_create):
			self.__dr = pylib.dimreducers_collection.DimReducerCollection\
				.from_serialzed(self.dimreducer_props)
			self.reset_eval_results("all")
			# model maybe huge, always force gc after new creation
			# this collects the old model if it is unreachable
			gc.collect()
		return self.__dr

	def get_classifier(self, *, force_create = False):
		"""
		get the classifier object;
		if not initialized, or force_create = True, create and return a new one;
		NOTE: create a new object in any case will reset all evaluation results;
		"""
		if ((self.__cf is None) or force_create):
			self.__cf = pylib.classifiers_collection.ClassifierCollection\
				.from_serialzed(self.classifier_props)
			self.reset_eval_results("all")
			# model maybe huge, always force gc after new creation
			# this collects the old model if it is unreachable
			gc.collect()
		return self.__cf

	############################################################################
	# .fit(), .predict(), .evaluate() are abstract methods defined in base class
	# ModelStructureAbstract
	def fit(self, X, Y, *, dimreducer_args = dict(), classifier_args = dict(),
			force_create = False):
		"""
		fit the model complex by first train the dimension reduction method then
		perform classifier training;
		NOTE: fit the model will reset testing evaluation results;

		ARGUMENT
		X, Y: follow .fit() conventions
		dimreducer_args: extra arguments passed to dimreducer.fit()
		classifier_args: extra arguments passed to classifier.fit()
		force_create: True if always create new dimreducer and classifier to
			replace old ones
		"""
		dr = self.get_dimreducer(force_create = force_create)
		cf = self.get_classifier(force_create = force_create)
		tr_x = dr.fit_transform(X, Y, **dimreducer_args)
		cf.fit(tr_x, Y)
		# update training evaluation
		self.set_eval_results("training", pylib.evaluator.ClassifEvaluator\
			.evaluate(true_label = Y, pred_label = cf.predict(tr_x)))
		# clear testing evaluation after (re)fitting
		self.reset_eval_results("testing")
		return self

	def predict(self, X, Y = None):
		"""
		predict the classified labels using trained model; the input X will be
		first transformed (reduced dimensionality) by dimreducer before passed
		to classifier.predict();

		ARGUMENT
		X: follows .predict() convention;
		Y: if provided, consider as true labels and performs model evaluation;
		"""
		dr = self.get_dimreducer(force_create = False)
		cf = self.get_classifier(force_create = False)
		tr_x = dr.transform(X)
		ret = cf.predict(tr_x)
		# if provided Y, calculate testing accuracy, etc.
		if Y is not None:
			self.set_eval_results("testing", pylib.evaluator.ClassifEvaluator\
				.evaluate(true_label = Y, pred_label = ret))
		return ret

	############################################################################
	# .serialize() and .deserialze() are protocol of SerializerAbstract
	def serialize(self):
		ret = dict(evaluation = self.get_eval_results("all"),
			dimreducer = self.get_dimreducer().serialize(),
			classifier = self.get_classifier().serialize())
		return ret

	@classmethod
	def deserialze(cls, ds):
		# evaluation will not be deserialzed
		new = cls(dimreducer_props = ds["dimreducer"],
			classifier_props = ds["classifier"])
		return new
