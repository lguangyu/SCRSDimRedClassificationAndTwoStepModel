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

	# level 2 parameters
	gp = ap.add_argument_group("level 1 options")
	gp.add_argument("-R1", "--level1-dimreducer", type = str, required = True,
		metavar = "model",
		help = "dimension reduction method in level 1 (required); choices: "\
			+ (", ".join(pylib.DimReducerCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")
	gp.add_argument("-D1", "--level1-reduce-dim-to", type = int,
		required = True, metavar = "int",
		help = "reduce dimensionality to this value in the level 1, must be "
			"positive (required)")
	gp.add_argument("-C1", "--level1-classifier", type = str, required = True,
		metavar = "model",
		choices = pylib.ClassifierCollection.get_registered_keys(),
		help = "classifier method in level (required); choices: "\
			+ (", ".join(pylib.ClassifierCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")

	# level 2 parameters
	gp = ap.add_argument_group("level 2 options")
	gp.add_argument("-R2", "--level2-dimreducer", type = str, required = True,
		metavar = "model",
		help = "dimension reduction method in level 2 (required); choices: "\
			+ (", ".join(pylib.DimReducerCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")
	gp.add_argument("-D2", "--level2-reduce-dim-to", type = int,
		required = True, metavar = "int",
		help = "reduce dimensionality to this value in the level 2, must be "
			"positive (required)")
	gp.add_argument("-C2", "--level2-classifier", type = str, required = True,
		metavar = "model",
		choices = pylib.ClassifierCollection.get_registered_keys(),
		help = "classifier method in level (required); choices: "\
			+ (", ".join(pylib.ClassifierCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")
	gp.add_argument("--indep-level2", action = "store_true",
		help = "train level2 model independently from output of level 1 model "\
			+ "(default: off)")

	# cross validation parameters
	gp = ap.add_argument_group("cross-validation options")
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
	# check args
	# ckeck dimension reduction
	if args.level1_reduce_dim_to <= 0:
		raise ValueError("--level1-reduce-dim-to must be positive")
	if args.level2_reduce_dim_to <= 0:
		raise ValueError("--level2-reduce-dim-to must be positive")
	return args


def main():
	args = get_args()
	# check and create dataset
	if not pylib.DatasetCollection.check_query_subclass(args.dataset,
		exp_cls = pylib.dataset.DuoLabelDatasetBase):
		raise RuntimeError("dataset used by 2-level method must be "
			"duo-labelled (both phase and strain), '%s' is incompatible"\
			% args.dataset)
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
	# create cross validator
	cv = pylib.model_structures.CrossValidator(
		cv_props = dict(n_splits = args.cv_folds, **args.cv_shuffle))
	# create model
	lv1_props = {
		"dimreducer_props": {
			"model": args.level1_dimreducer,
			"params": {"n_components": args.level1_reduce_dim_to},
		},
		"classifier_props": {
			"model": args.level1_classifier,
			"params": None,
		},
	}
	lv2_props = {
		"dimreducer_props": {
			"model": args.level2_dimreducer,
			"params": {"n_components": args.level2_reduce_dim_to},
		},
		"classifier_props": {
			"model": args.level2_classifier,
			"params": None,
		},
	}
	model = pylib.model_structures.TwoLevelDimredClassifEnsemble(
		lv1_props = lv1_props, lv2_props = lv2_props,
		indep_lv2 = args.indep_level2)
	# run cv
	cv.cross_validate(model, X = dataset.data,
		Y = (dataset.phase_label, dataset.strain_label),
		duo_label = True)
	# output
	out = dict(mode = "2-level", dataset = args.dataset,
		phase_labels = dataset.phase_label_encoder.classes_.tolist(),
		strain_labels = dataset.strain_label_encoder.classes_.tolist(),
		results = cv.get_cv_results())
	with pylib.util.file_io.get_fh(args.output, "w") as fp:
		json.dump(out, fp, sort_keys = True,
			cls = pylib.util.serializer.SerializerJSONEncoder)
	return


if __name__ == "__main__":
	main()
