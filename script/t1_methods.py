#!/usr/bin/env python3

import os
import sys
import numpy
import argparse
import sklearn.preprocessing
import custom_pylib.cross_validation


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--data", type = str, metavar = "tsv", required = True,
		help = "sample data file tsv (required)")
	ap.add_argument("-m", "--meta", type = str, metavar = "tsv", required = True,
		help = "metadata file tsv (required)")
	ap.add_argument("-c", "--classifier", type = str, required = True,
		choices = ["gnb", "lr", "lda", "svm_lin", "svm_rbf"],
		help = "choice of fitting model (required)")
	ap.add_argument("-f", "--cv-folds", type = int, metavar = "int", default = 10,
		help = "n-fold cross validation (default: 10)")
	ap.add_argument("-p", "--permutation", type = str,
		metavar = "disable|random|int", default = "disable",
		help = "permutate samples, can be disable, random or specify a seed (int) (default: disable)")
	ap.add_argument("-R", "--dim-reduc", type = str,
		default = "none", choices = ["none", "pca", "lda", "lsdr"],
		help = "choosing dimension reduction method (default: none)")
	ap.add_argument("-D", "--reduce-dim-to", type = str,
		metavar = "none|int", default = "none",
		help = "reduce dimensionality to this value, must be positive and is required if --dim-reduc is not 'none', omitted otherwise")
	gp = ap.add_argument_group("output")
	gp.add_argument("--output-txt-dir", type = str, metavar = "dir", default = "output",
		help = "output txt results to this dir (default: output)")
	gp.add_argument("--output-png-dir", type = str, metavar = "dir", default = "image",
		help = "output png results to this dir (default: image)")
	args = ap.parse_args()
	# check args
	_check_args_cv_folds(args)
	_check_args_dim_reduction(args)
	_check_args_permutation(args)
	# format output fname string
	# excluding folder and extension
	args.output_str = "%s.%s.dr_%s_%s.%d_fold" % (
		os.path.basename(args.data), args.classifier,
		args.dim_reduc, str(args.reduce_dim_to), args.cv_folds)
	return args


def _check_args_cv_folds(args):
	if args.cv_folds <= 0:
		raise ValueError("--cv-folds must be positive")
	return


def _check_args_dim_reduction(args):
	if args.dim_reduc == "none":
		args.reduce_dim_to = "none" # set this to none as omitted
	else:
		if args.reduce_dim_to == "none":
			raise ValueError("--reduce-dim-to is required when --dim-reduc not none")
		# turn string into int
		args.reduce_dim_to = int(args.reduce_dim_to)
		if args.reduce_dim_to <= 0:
			raise ValueError("--reduce-dim-to must be positive")
	return


def _check_args_permutation(args):
	if args.permutation not in ["disable", "random"]:
		# try setting as seed
		args.permutation = int(args.permutation)
	return


def print_runinfo(args, fh = sys.stderr):
	arg_vars = vars(args)
	for key in ["data", "meta", "classifier", "cv_folds",
		"permutation", "dim_reduc", "reduce_dim_to"]:
		print(key + ":", arg_vars[key], file = fh)


def load_data(fdata, fmeta):
	data = numpy.loadtxt(fdata, dtype = float, delimiter = "\t")
	with open(fmeta, "r") as fh:
		meta = fh.read().splitlines()
	# label is the 1st col in metadata file
	labels = [i.split("\t")[0] for i in meta]
	labels = numpy.asarray(labels, dtype = object)
	return data, labels


def get_unique_labels(labels):
	uniques = numpy.unique(labels)
	uniques.sort()
	return uniques


def encode_labels(labels):
	encoder = sklearn.preprocessing.LabelEncoder()
	encoder.fit(labels)
	return encoder, encoder.transform(labels)


def main():
	args = get_args()
	print_runinfo(args)

	# load data
	data, labels = load_data(args.data, args.meta)

	# preprocessing
	uniq_labels = get_unique_labels(labels)
	lab_encoder, encoded_labels = encode_labels(labels)

	# cross validation
	cv = custom_pylib.cross_validation.CrossValidation(
		classifier = args.classifier, n_fold = args.cv_folds,
		permutation = ("random" if args.permutation else "disable"),
		dim_reduc = args.dim_reduc, reduce_dim_to = args.reduce_dim_to)
	cv.run_cv(data, encoded_labels)
	cv_res = cv.get_result()

	# save text result
	txt_prefix = os.path.join(args.output_txt_dir, args.output_str)
	cv.savetxt(txt_prefix, uniq_labels, lab_encoder)
	
	# save plots
	png_prefix = os.path.join(args.output_png_dir, args.output_str)
	cv.savetxt(txt_prefix, uniq_labels, lab_encoder)
	boxplot_title = "%s, classifier=%s, dr=%s(%s), n_fold=%d" \
		% (os.path.basename(args.data), args.classifier,
		args.dim_reduc, str(args.reduce_dim_to), args.cv_folds)
	cv.savefig_boxplot(png_prefix, uniq_labels, lab_encoder,
		title = boxplot_title)
	return


if __name__ == "__main__":
	main()
