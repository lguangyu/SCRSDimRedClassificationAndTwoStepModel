#!/usr/bin/env python3

import argparse
import json
import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "-d", "--dataset", type = str, required = True,
		metavar = "dataset",
		choices = pylib.DatasetCollection.get_registered_keys(),
		help = "the dataset to run model on (required); choices: "\
			+ (", ".join(pylib.DatasetCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "json",
		help = "write output to this file instead of stdout")

	# model parameters
	gp = ap.add_argument_group(description = "model parameters")
	gp.add_argument("-R", "--dimreducer", type = str, required = True,
		metavar = "model",
		choices = pylib.DimReducerCollection.get_registered_keys(),
		help = "dimension reduction method (required); choices: "\
			+ (", ".join(pylib.DimReducerCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")
	gp.add_argument("-D", "--reduce-dim-to", type = int, required = True,
		metavar = "int",
		help = "reduce dimensionality to this value, must be positive "
			"(required)")
	gp.add_argument("-C", "--classifier", type = str, required = True,
		metavar = "model",
		choices = pylib.ClassifierCollection.get_registered_keys(),
		help = "classifier method (required); choices: "\
			+ (", ".join(pylib.ClassifierCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")

	# cross validation parameters
	gp = ap.add_argument_group(description = "cross-validation parameters")
	gp.add_argument("-f", "--cv-folds", type = int,
		metavar = "int", default = 10,
		help = "n-fold cross validation, must be at least 2 (default: 10)")
	gp.add_argument("--cv-shuffle",
		type = pylib.util.arg_parsing.CVShuffleParam,
		default = pylib.util.arg_parsing.CVShuffleParam("random"),
		metavar = "random|int|disable",
		help = "set random permuation state of the cross-validator invoked; "
			"random = randomly shuffle samples for each .split() call; "
			"int = use specific random seed; "
			"disable = completely disable shuffling; (default: random)")

	args = ap.parse_args()
	# refine args
	if args.output == "-":
		args.output = sys.stdout
	return args


def main():
	args = get_args()
	# create dataset
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
	if not isinstance(dataset, pylib.dataset.SingleLabelDataset):
		raise ValueError("dataset used in this (1-level) model must be "
			"SingleLabelDataset, but the specified '%s' is not"\
			% args.dataset)
	# create cross-validator
	cv = pylib.model_structures.CrossValidator(
		cv_props = dict(n_splits = args.cv_folds, **args.cv_shuffle))
	# create model
	dimreducer_props = {"model": args.dimreducer,
		"params": {"n_components": args.reduce_dim_to}}
	classifier_props = {"model": args.classifier, "params": None}
	model = pylib.model_structures.DimredClassifComplex(
		dimreducer_props = dimreducer_props,
		classifier_props = classifier_props)
	# run cv
	cv.cross_validate(model, X = dataset.data, Y = dataset.label)
	# output
	out = dict(mode = "1-level",
		labels = dataset.label_encoder.classes_.tolist(),
		results = cv.get_cv_results())
	with pylib.util.file_io.get_fh(args.output, "w") as fp:
		json.dump(out, fp, sort_keys = True,
			cls = pylib.util.serializer.SerializerJSONEncoder)
	return


if __name__ == "__main__":
	main()
