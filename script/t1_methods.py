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
		help = "the dataset to run model on (required); choices: %s"\
			% pylib.DatasetCollection.repr_reg_keys())
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "json",
		help = "write output to this file instead of stdout")
	ap.add_argument("--human-readable", action = "store_true",
		help = "save output in human-readable form, will take slightly more "
			"space (default: no)")

	# model parameters
	gp = ap.add_argument_group("model options")
	gp.add_argument("-R", "--dimreducer", type = str, required = True,
		metavar = "model",
		choices = pylib.DimReducerCollection.get_registered_keys(),
		help = "dimension reduction method (required); choices: %s"\
			% pylib.DimReducerCollection.repr_reg_keys())
	gp.add_argument("-D", "--reduce-dim-to", type = int, required = True,
		metavar = "int",
		help = "reduce dimensionality to this value, must be positive "
			"(required)")
	gp.add_argument("-C", "--classifier", type = str, required = True,
		metavar = "model",
		choices = pylib.ClassifierCollection.get_registered_keys(),
		help = "classifier method (required); choices: %s"\
			% pylib.ClassifierCollection.repr_reg_keys())

	# cross validation parameters
	gp = ap.add_argument_group("cross-validation options")
	gp.add_argument("-f", "--cv-folds", type = int,
		metavar = "int", default = 10,
		help = "n-fold cross validation, must be at least 2 (default: 10)")
	gp.add_argument("-j", "--cv-parallel", type = int,
		metavar = "int", default = 1,
		help = "parallel jobs to run cross-validation (default: 1)")
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
	# check and create dataset
	if not pylib.DatasetCollection.check_query_subclass(args.dataset,
		exp_cls = pylib.dataset.SingleLabelDatasetBase):
		raise RuntimeError("dataset used by 1-level method must be "
			"single-labelled (only strain or phase), '%s' is incompatible"\
			% args.dataset)
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
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
	cv.cross_validate(model, X = dataset.data, Y = dataset.label,
		n_jobs = args.cv_parallel)
	# output
	out = dict(mode = "1-level", dataset = args.dataset,
		labels = dataset.label_encoder.classes_.tolist(),
		results = cv.get_cv_results())
	pylib.util.file_io.save_as_json(out, args.output,
			human_readable = args.human_readable,
			cls = pylib.util.serializer.SerializerJSONEncoder)
	return


if __name__ == "__main__":
	main()
