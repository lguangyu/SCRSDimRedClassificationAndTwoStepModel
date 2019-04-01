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


def get_args():
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
	ap.add_argument("-D1", "--level1-reduce-dim-to", type = int, required = True,
		metavar = "int",
		help = "reduce dimension to this value in the 1st level (required)")
	ap.add_argument("-D2", "--level2-reduce-dim-to", type = int, required = True,
		metavar = "int",
		help = "reduce dimension to this value in the 1st level (required)")
	ap.add_argument("--indep-levels", action = "store_true",
		help = "train level2 model independently from output of level1 model "\
			+ "(default: off)")

	gp = ap.add_argument_group("output")
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
	#print_runinfo(args)
	# load data
	data, lv1_labels, lv2_labels = load_data(args.config)
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
	# output
	os.makedirs(args.output_txt_dir, exist_ok = True)
	txt_output = os.path.join(args.output_txt_dir,
		"%s.two_level.%s_fold.txt"\
			% (os.path.basename(args.config), str(args.cv_folds)))
	with open(txt_output, "w") as fh:
		print("\t".join(["lv1-labels:"] + list(lv1_label_encoder.classes_)),\
			file = fh)
		print("\t".join(["lv2-labels:"] + list(lv2_label_encoder.classes_)),\
			file = fh)
		# model
		dr_cls_comb = itertools.product(\
			pylib.dim_reducer.list_registered(),\
			pylib.classifier.list_registered())
		for (lv1_dr, lv1_cls), (lv2_dr, lv2_cls) in itertools.product(\
			dr_cls_comb, repeat = 2):
			# level 2 model only does dimension reduction enabled combinations
			if lv1_dr == "none" or lv2_dr == "none":
				continue
			# create model
			model = pylib.TwoLevelModel(\
				level1_props = dict(\
					dim_reducer = lv1_dr,\
					classifier = lv1_cls,\
					dims_remain = args.level1_reduce_dim_to),
				level2_props = dict(\
					dim_reducer = lv2_dr,\
					classifier = lv2_cls,\
					dims_remain = args.level2_reduce_dim_to),
				indep_level2 = args.indep_levels)
			# cross validation
			cv = pylib.TwoLevelCrossValidator(model, args.cv_folds, args.permutation)
			# run cv
			cv.run_cv(data, encoded_univ_labels, encoded_lv1_labels, encoded_lv2_labels)
			# output
			print("model:\t%s+%s/%s+%s" % (lv1_dr, lv1_cls, lv2_dr, lv2_cls),\
				file = fh)
			for i in range(args.cv_folds):
				print("fold %d, training evaluation:" % (i + 1), file = fh)
				cv.train_evaluation[i].dump_txt(fh)
				print("fold %d, testing evaluation:" % (i + 1), file = fh)
				cv.test_evaluation[i].dump_txt(fh)
	return


if __name__ == "__main__":
	main()
