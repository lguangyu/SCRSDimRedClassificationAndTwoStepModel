#!/usr/bin/env python3

import sys
import os
import argparse
import numpy


SUMMARIZE_DATASETS = [
	"EXPONENT1-50.normalized_l2.data.tsv",
	"PLATFORM1-50.normalized_l2.data.tsv",
	"PLATFORM2-50.normalized_l2.data.tsv",
]

SUMMARIZE_CLASSIFIERS = [
	dict(id = "gnb", display_name = "GNB"),
	dict(id = "lr", display_name = "LR"),
	dict(id = "lda", display_name = "LDA"),
	dict(id = "svm_lin", display_name = "Linear SVM"),
	dict(id = "svm_rbf", display_name = "RBF SVM"),
]

SUMMARIZE_DIMREDUC = [
	dict(id = "none_none",	display_name = "N/A"),
	dict(id = "pca_26",		display_name = "PCA(26)"),
	dict(id = "lda_26",		display_name = "LDA(26)"),
	dict(id = "lsdr_26",	display_name = "LSDR(26)"),
]


class ResultNotFoundInFileError(RuntimeError):
	def __init__(self, *ka, **kw):
		super(ResultNotFoundInFileError, self).__init__(*ka, **kw)


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-dir",
		type = str, metavar = "dir", required = True,
		help = "directory of cross validation results (required)")
	ap.add_argument("-d", "--delimiter",
		type = str, metavar = "char", default = "\t",
		help = "delimiter in input/output file (default: <tab>)")
	ap.add_argument("-o", "--output-dir",
		type = str, metavar = "dir", default = ".",
		help = "output directory (default: .)")
	args = ap.parse_args()
	return args


def get_results(fname, category, delimiter = "\t"):
	with open(fname, "r") as fh:
		lines = fh.read().splitlines()
	for l in lines:
		l = l.split(delimiter)
		if l[0] == category:
			return numpy.asarray(l[1:], dtype = float)
	raise ResultNotFoundInFileError("no '%s' found in file '%s'"\
		% (category, fname))
	return


def summarize_dataset_results(args, dataset):
	ofile = os.path.join(args.output_dir, "%s.summary.tsv" % dataset)
	with open(ofile, "w") as fh:
		col_headers = list(map(lambda i: i["display_name"], SUMMARIZE_DIMREDUC))
		header = (args.delimiter).join([""] + col_headers)
		print(header, file = fh)
		#
		for cls in SUMMARIZE_CLASSIFIERS:
			# each line is a classifier
			line = [cls["display_name"]]
			for dr in SUMMARIZE_DIMREDUC:
				# fina reaults for dim redcution methods
				fname = os.path.join(args.input_dir,
					"%s.%s.dr_%s.10_fold.overall.txt"\
					% (dataset, cls["id"], dr["id"]))
				res = get_results(fname, "accuracy", args.delimiter)
				formatted = "%.3f (sd: %.3f)" % (res.mean(), res.std())
				line.append(formatted)
			print((args.delimiter).join(line), file = fh)
		return


def main():
	args = get_args()
	for dataset in SUMMARIZE_DATASETS:
		summarize_dataset_results(args, dataset)
	return


if __name__ == "__main__":
	main()
