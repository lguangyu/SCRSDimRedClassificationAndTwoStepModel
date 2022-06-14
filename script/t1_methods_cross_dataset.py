#!/usr/bin/env python3

import argparse
import json
import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--training-dataset", type = str,
		required = True, metavar = "dataset",
		choices = pylib.DatasetCollection.get_registered_keys(),
		help = "the dataset to train model on (required); choices: %s"\
			% pylib.DatasetCollection.repr_reg_keys())
	ap.add_argument("-t", "--testing-dataset", type = str,
		required = True, metavar = "dataset",
		choices = pylib.DatasetCollection.get_registered_keys(),
		help = "the dataset to test model on (required); choices: %s"\
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

	args = ap.parse_args()
	# refine args
	if args.output == "-":
		args.output = sys.stdout
	return args


def main():
	args = get_args()
	# check and create dataset
	if not pylib.DatasetCollection.check_query_subclass(args.training_dataset,
		exp_cls = pylib.dataset.SingleLabelDatasetBase):
		raise RuntimeError("training dataset used by 1-level method must be "
			"single-labelled (only strain or phase), '%s' is incompatible"\
			% args.training_dataset)
	if not pylib.DatasetCollection.check_query_subclass(args.testing_dataset,
		exp_cls = pylib.dataset.SingleLabelDatasetBase):
		raise RuntimeError("testing dataset used by 1-level method must be "
			"single-labelled (only strain or phase), '%s' is incompatible"\
			% args.testing_dataset)
	training_dataset = pylib.DatasetCollection.get_dataset(args.training_dataset)
	testing_dataset = pylib.DatasetCollection.get_dataset(args.testing_dataset)
	# create model
	dimreducer_props = {"model": args.dimreducer,
		"params": {"n_components": args.reduce_dim_to}}
	classifier_props = {"model": args.classifier, "params": None}
	model = pylib.model_structures.DimredClassifComplex(
		dimreducer_props = dimreducer_props,
		classifier_props = classifier_props)
	# train model using training dataset
	model.fit(training_dataset.data, training_dataset.label)
	model.predict(testing_dataset.data, testing_dataset.label)
	# output
	out = dict(mode = "1-level-cross-dataset",
		training_dataset = args.training_dataset,
		testing_dataset = args.testing_dataset,
		labels = training_dataset.label_encoder.classes_.tolist(),
		results = model,
	)
	pylib.util.file_io.save_as_json(out, args.output,
		human_readable = args.human_readable,
		cls = pylib.util.serializer.SerializerJSONEncoder)
	return


if __name__ == "__main__":
	main()
