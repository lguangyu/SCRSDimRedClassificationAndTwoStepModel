#!/usr/bin/env python3

import argparse
import itertools
import json
import numpy
import os
import sklearn.model_selection
import sklearn.preprocessing
import sys
import pylib


class CommaSepList(object):
	"""
	split s into a list by using comma as separator
	if value_set in __init__ is not None, also checks if all splitted values
	belong to the value_set
	"""
	@property
	def value_set(self):
		return self._value_set
	@value_set.setter
	def value_set(self, value):
		if (value is None) or (isinstance(value, list)):
			self._value_set = value
			return
		raise TypeError("value_set must be None or list")

	def __init__(self, value_set: list = None):
		self.value_set = value_set
		return 

	def __call__(self, s: str):
		splitted = s.split(",")
		if self.value_set is not None:
			for i in splitted:
				if i not in self.value_set:
					raise ValueError("%s not in choices %s"\
						% (i, str(self.value_set)))
		return splitted

	def __repr__(self):
		if self.value_set is None:
			return "CommaSepList(any)"
		else:
			return "CommaSepList(choices: %s)" % (",".join(self.value_set))


def get_args():
	DRS = pylib.dim_reducer.list_registered()
	CLS = pylib.classifier.list_registered()
	#
	ap = argparse.ArgumentParser()
	ap.add_argument("config", type = str,
		help = "running config json file")
	ap.add_argument("-f", "--cv-folds", type = int,
		metavar = "int", default = 10,
		help = "n-fold cross validation, must be at least 2 (default: 10)")
	ap.add_argument("-p", "--permutation", type = str,
		metavar = "disable|random|int", default = "random",
		help = "permutate samples, can be 'disable', 'random' "\
			+ "or an integer as seed (default: random)")
	#
	gp = ap.add_argument_group("level 1 settings")
	gp.add_argument("-R1", "--level1-dim-reduc", type = CommaSepList(DRS),
		metavar = "...", default = DRS,
		help = "choices of dimension reducers in level 1, comma separated "\
			+ ("(default: %s)" % (",".join(DRS))))
	gp.add_argument("-D1", "--level1-reduce-dim-to", type = int,
		required = True, metavar = "int",
		help = "reduce dimension to this value in the level 1 (required)")
	gp.add_argument("-C1", "--level1-classifier", type = CommaSepList(CLS),
		metavar = "...", default = CLS,
		help = "choices of classifiers in level 1, comma separated "\
			+ ("(default: %s)" % (",".join(CLS))))
	#
	gp = ap.add_argument_group("level 2 settings")
	gp.add_argument("-R2", "--level2-dim-reduc", type = CommaSepList(DRS),
		metavar = "...", default = DRS,
		help = "choices of dimension reducers in level 2, comma separated "\
			+ ("(default: %s)" % (",".join(DRS))))
	gp.add_argument("-D2", "--level2-reduce-dim-to", type = int,
		required = True, metavar = "int",
		help = "reduce dimension to this value in the level 2 (required)")
	gp.add_argument("-C2", "--level2-classifier", type = CommaSepList(CLS),
		metavar = "...", default = CLS,
		help = "choices of classifiers in level 2, comma separated "\
			+ ("(default: %s)" % (",".join(CLS))))
	gp.add_argument("--indep-level2", action = "store_true",
		help = "train level2 model independently from output of level1 model "\
			+ "(default: off)")
	#
	gp = ap.add_argument_group("output settings")
	gp.add_argument("--output-txt-dir", type = str,
		metavar = "dir", default = "output",
		help = "output txt results to this dir (default: output)")
	gp.add_argument("--output-png-dir", type = str,
		metavar = "dir", default = "image",
		help = "output png results to this dir (default: image)")
	args = ap.parse_args()
	# check args
	# ckeck dimension reduction
	if args.level1_reduce_dim_to <= 0:
		raise ValueError("--level1-reduce-dim-to must be positive")
	if args.level2_reduce_dim_to <= 0:
		raise ValueError("--level2-reduce-dim-to must be positive")
	return args


def load_data(config) -> "data, label":
	with open(config, "r") as fh:
		cfg = json.load(fh)
	# load data
	data = numpy.loadtxt(cfg["data_file"], dtype = float, delimiter = "\t")
	# load labels
	with open(cfg["label_file"], "r") as fh:
		lines = fh.read().splitlines()
		lines = [i.split("\t") for i in lines]
		lv1_labels = [line[cfg["level1_label_col"]] for line in lines]
		lv2_labels = [line[cfg["level2_label_col"]] for line in lines]
	assert len(data) == len(lv1_labels), "%d|%d" % (len(data), len(lv1_labels))
	assert len(data) == len(lv2_labels), "%d|%d" % (len(data), len(lv2_labels))
	return data, lv1_labels, lv2_labels


def main():
	args = get_args()
	# load data
	data, lv1_labels, lv2_labels = load_data(args.config)
	data = sklearn.preprocessing.scale(data)
	assert numpy.isclose(numpy.mean(data, axis = 0), 0).all()
	assert numpy.isclose(numpy.std(data, axis = 0), 1).all()
	univ_labels = [i + j for i, j in zip(lv1_labels, lv2_labels)]
	# preprocessing, encode labels
	univ_label_encoder = sklearn.preprocessing.LabelEncoder()
	univ_label_encoder.fit(univ_labels)
	encoded_univ_labels = univ_label_encoder.transform(univ_labels)
	#
	lv1_label_encoder = sklearn.preprocessing.LabelEncoder()
	lv1_label_encoder.fit(lv1_labels)
	encoded_lv1_labels = lv1_label_encoder.transform(lv1_labels)
	#
	lv2_label_encoder = sklearn.preprocessing.LabelEncoder()
	lv2_label_encoder.fit(lv2_labels)
	encoded_lv2_labels = lv2_label_encoder.transform(lv2_labels)
	# results
	results = {}
	results["lv1_labels"] = list(lv1_label_encoder.classes_)
	results["lv2_labels"] = list(lv2_label_encoder.classes_)
	results["models"] = {}
	# model
	for lv1_dr, lv1_cls, lv2_dr, lv2_cls in itertools.product(
		args.level1_dim_reduc, args.level1_classifier,
		args.level2_dim_reduc, args.level2_classifier):
		# log running model
		model_name = "%s+%s/%s+%s" % (lv1_dr, lv1_cls, lv2_dr, lv2_cls)
		# create model
		try:
			model = pylib.TwoLevelModel(\
				level1_props = dict(\
					dim_reducer = lv1_dr,\
					classifier = lv1_cls,\
					dims_remain = args.level1_reduce_dim_to),
				level2_props = dict(\
					dim_reducer = lv2_dr,\
					classifier = lv2_cls,\
					dims_remain = args.level2_reduce_dim_to),
				indep_level2 = args.indep_level2)
			# cross validation
			cv = pylib.TwoLevelCrossValidator(model, args.cv_folds, args.permutation)
			# run cv
			cv.run_cv(data, encoded_univ_labels, encoded_lv1_labels, encoded_lv2_labels)
			# results output
			results["models"][model_name] = cv.evaluation.copy()
		except Exception as e:
			results["models"][model_name] = "error:" + repr(e)
	# output
	os.makedirs(args.output_txt_dir, exist_ok = True)
	txt_output = os.path.join(args.output_txt_dir,
		"%s.%d_fold.%s.json"\
			% (os.path.basename(args.config), args.cv_folds,\
			"lv2_indep" if args.indep_level2 else "lv2_dep"))
	with open(txt_output, "w") as fh:
		json.dump(results, fh)
	return


if __name__ == "__main__":
	main()
