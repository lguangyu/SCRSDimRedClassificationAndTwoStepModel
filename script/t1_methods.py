#!/usr/bin/env python3

import argparse
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
	#
	gp = ap.add_argument_group("model settings")
	gp.add_argument("-R", "--dim-reduc", type = str,
		default = "none", choices = pylib.dim_reducer.list_registered(),
		help = "choice of dimension reduction method (default: none)")
	gp.add_argument("-D", "--reduce-dim-to", type = str,
		metavar = "none|int", default = "none",
		help = "reduce dimensionality to this value, must be positive "\
			+ "and is required if --dim-reduc is not 'none', omitted otherwise")
	gp.add_argument("-C", "--classifier", type = str, required = True,
		choices = pylib.classifier.list_registered(),
		help = "choice of classifier (required)")
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
	if args.dim_reduc == "none":
		args.reduce_dim_to = None # set this to none as omitted
	else:
		if args.reduce_dim_to == "none":
			raise ValueError("--reduce-dim-to is required when --dim-reduc not none")
		# turn string into int
		args.reduce_dim_to = int(args.reduce_dim_to)
		if args.reduce_dim_to <= 0:
			raise ValueError("--reduce-dim-to must be positive")
	return args


def print_runinfo(args, fh = sys.stdout):
	arg_vars = vars(args)
	for key in ["data", "meta", "classifier", "cv_folds",
		"permutation", "dim_reduc", "reduce_dim_to"]:
		print(key + ":", arg_vars[key], file = fh)


def load_data(config) -> "data, label":
	with open(config, "r") as fh:
		cfg = json.load(fh)
	# load data
	data = numpy.loadtxt(cfg["data_file"], dtype = float, delimiter = "\t")
	# load labels
	with open(cfg["label_file"], "r") as fh:
		labels = [line.replace("\n", "").split("\t")[cfg["label_col"]]\
			for line in fh]
	assert len(data) == len(labels), "%d|%d" % (len(data), len(labels))
	return data, labels


def main():
	args = get_args()
	#print_runinfo(args)
	# load data
	data, labels = load_data(args.config)
	data = sklearn.preprocessing.scale(data)
	assert numpy.isclose(numpy.mean(data, axis = 0), 0).all()
	assert numpy.isclose(numpy.std(data, axis = 0), 1).all()
	# preprocessing, encode labels
	label_encoder = sklearn.preprocessing.LabelEncoder()
	label_encoder.fit(labels)
	encoded_labels = label_encoder.transform(labels)
	# model
	model = pylib.SingleLevelModel(\
		dim_reducer = args.dim_reduc,\
		classifier = args.classifier,\
		dims_remain = args.reduce_dim_to)
	# FIXME: this is temporary
	if args.dim_reduc == "lsdr_reg":
		model.regularizer_list = [0, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
	# cross validation
	cv = pylib.SingleLevelCrossValidator(model, args.cv_folds, args.permutation)
	# run cv
	cv.run_cv(data, encoded_labels)

	# results
	results = {}
	results["class_labels"] = list(label_encoder.classes_)
	results["folds"] = cv.evaluation.copy()
	# output
	os.makedirs(args.output_txt_dir, exist_ok = True)
	txt_output = os.path.join(args.output_txt_dir,
		"%s.%s.dr_%s_%s.%d_fold.txt" % (\
			os.path.basename(args.config),
			args.classifier,
			args.dim_reduc, args.reduce_dim_to,
			args.cv_folds))
	with open(txt_output, "w") as fh:
		json.dump(results, fh)
	## save plots
	#png_prefix = os.path.join(args.output_png_dir, args.output_str)
	#boxplot_title = "%s, classifier=%s, dr=%s(%s), n_fold=%d"\
	#	% (os.path.basename(args.data), args.classifier,
	#	args.dim_reduc, str(args.reduce_dim_to), args.cv_folds)
	#cv.savefig_boxplot(png_prefix, uniq_labels, lab_encoder,
	#	title = boxplot_title)
	return


if __name__ == "__main__":
	main()
