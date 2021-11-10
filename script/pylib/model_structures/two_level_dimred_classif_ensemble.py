#!/usr/bin/env python3

import functools
import gc
import numpy
# custom lib
from . import base
from . import dimreduc_classif_complex
import pylib.evaluator


class TwoLevelDimredClassifEnsemble(pylib.evaluator.ModelEvaluationResultsMixin,
		base.ModelStructureAbstract):
	"""
	an multi-layer ensemble model structure for sequentially training dimension
	reducer and classifier complexes with two sets of labels; the first label is
	used to train the first layer of models, and the second layer is trained by
	the second set of labels; each level uses a dimension reducer and classifier
	complex, i.e. DimredClassifComplex();

	ARGUMENT
	lv1_props: arguments used to create lv1 DimredClassifComplex model(s);
	lv2_props: arguments used to create lv2 DimredClassifComplex model(s);
	indep_lv2: if True, the lv2 model will be trained independent to lv1 output;
		if False, each of the lv2 model will be trained per scope of each
		unique label output from the lv1 models; default if False;
	"""
	def __init__(self, lv1_props: dict, lv2_props: dict, indep_lv2 = False,
			*ka, **kw):
		super(TwoLevelDimredClassifEnsemble, self).__init__(*ka, **kw)
		self.lv1_props = lv1_props
		self.lv2_props = lv2_props
		self.indep_lv2 = indep_lv2
		# lv1 and lv2 models, note lv2 is a series of models
		self.__lv1 = None
		self.__lv2 = None
		return

	############################################################################
	# per-level model creation, create_lv1() and create_lv2() will not force a
	# re-creation of models; re-creation will be assumed done buy model objects
	def get_submodel(self, model_props):
		try:
			ret = dimreduc_classif_complex.DimredClassifComplex(**model_props)
		except Exception as e:
			raise ValueError("model_props must be compatible with "
				"DimredClassifComplex(**model_props) constructor call; check "
				"DimredClassifComplex.__init__() for more information")\
				from e
		return ret

	def get_lv1(self, model_props = None):
		if self.__lv1 is None:
			if model_props is None:
				raise ValueError("init call requires model_props")
			self.__lv1 = self.get_submodel(model_props)
		return self.__lv1

	def get_lv2(self, model_props = None, n_estimators = None):
		if model_props is None:
			# not providing model_props intends to only query lv2 models, but no
			# creation
			if self.__lv2 is None:
				# this error since lv2 models are not yet created
				raise ValueError("init call requires model_props")
		else:
			# providing model_props intends to also create the models
			# first check n_estimators is also set
			# NOTE: n_estimators is ignored if model_props is None
			if n_estimators is None:
				raise ValueError("creation call requires both model_props and "
					"n_estimators")
			self.__lv2 = [self.get_submodel(model_props)\
				for i in range(n_estimators)]
		return self.__lv2

	############################################################################
	# .fit(), .predict(), .evaluate() are abstract methods defined in base class
	# ModelStructureAbstract
	def fit(self, X, Y, *, lv1_args = dict(), lv2_args = dict(),
			force_create = False):
		"""
		fit the model ensemble by first train the lv1 model complex using the
		first set of Y then the lv2 model complex using the second set of Y;
		NOTE: fit the model will reset testing evaluation results;

		ARGUMENT
		X: follow .fit() conventions
		Y: slightly different from conventional Y; the Y should be a list of
			sets of labels, i.e. [lv1_Y, lv2_Y]
		lv1_args: extra arguments parsed to lv1 models; should be competible
			with <lv1_model>.fit(**lv1_args)
		lv2_args: extra arguments parsed to lv2 models; should be competible
			with <lv2_model>.fit(**lv2_args)
		*NOTE* lv1_args and lv2_args can contain force_create argument if
			necessary
		"""
		# model creation/preparation
		lv1_labels, lv2_labels = Y
		lv1 = self.get_lv1(self.lv1_props)
		lv2 = self.get_lv2(self.lv2_props, len(numpy.unique(lv1_labels)))
		# fit lv1 model
		lv1.fit(X, lv1_labels, **lv1_args)
		# lv2 model delegation
		delg_labels = lv1_labels if self.indep_lv2 else lv1.predict(X)
		fit_pred = numpy.empty(len(X), dtype = int)
		for ulab in numpy.unique(delg_labels):
			ulab_mask = (delg_labels == ulab)
			lv2[ulab].fit(X[ulab_mask], lv2_labels[ulab_mask], **lv2_args)
			fit_pred[ulab_mask] = lv2[ulab].predict(X[ulab_mask])
		# update training evaluation
		self.set_eval_results("training-step-1",
			pylib.evaluator.ClassifEvaluator.evaluate(
				true_label = lv1_labels, pred_label = delg_labels))
		self.set_eval_results("training-step-2",
			pylib.evaluator.ClassifEvaluator.evaluate(
				true_label = lv2_labels, pred_label = fit_pred))
		# clear testing evaluation after (re)fitting
		self.reset_eval_results("testing-step-1")
		self.reset_eval_results("testing-step-2")
		return self

	def predict(self, X, Y = None):
		"""
		predict the classified labels using trained model; the input X will be
		first estimated lv1 label before making the final lv2 label prediction;

		ARGUMENT
		X: follows .predict() convention;
		Y: if provided, consider as true labels and performs model evaluation;
			Y should be only lv2 labels;
		"""
		lv1 = self.get_lv1()
		lv2 = self.get_lv2()
		ret = numpy.empty(len(X), dtype = int)
		lv1_pred = lv1.predict(X)
		# predict per lv1 output
		for ulab in numpy.unique(lv1_pred):
			ulab_mask = (lv1_pred == ulab)
			ret[ulab_mask] = lv2[ulab].predict(X[ulab_mask])
		# if provided Y, calculate testing accuracy, etc.
		if Y is not None:
			lv1_labels, lv2_labels = Y
			self.set_eval_results("testing-step-1",
				pylib.evaluator.ClassifEvaluator.evaluate(
					true_label = lv1_labels, pred_label = lv1_pred))
			self.set_eval_results("testing-step-2",
				pylib.evaluator.ClassifEvaluator.evaluate(
					true_label = lv2_labels, pred_label = ret))
		return ret

	############################################################################
	# .serialize() and .deserialize() are protocol of SerializerAbstract
	def serialize(self):
		ret = dict(evaluation = self.get_eval_results("all"),
			lv1_props = self.lv1_props,
			lv2_props = self.lv2_props,
			indep_lv2 = self.indep_lv2)
		return ret

	@classmethod
	def deserialize(cls, ds):
		# evaluation will not be deserialized
		new = cls(lv1_props = ds["lv1_props"], lv2_props = ds["lv2_props"],
			indep_lv2 = ds["indep_lv2"])
		return new
