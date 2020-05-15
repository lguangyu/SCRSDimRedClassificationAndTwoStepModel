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


def _make_level_props_dict(dim_reducer, classifier, dims_remain):
	ret = dict(dim_reducer = dim_reducer, classifier = classifier,\
		dims_remain = dims_remain)
	# FIXME: currently only recognizes lsdr_reg
	if dim_reducer == "lsdr_reg":
		ret["regularizer_list"] = [0, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
	return ret


def _main():
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
	res_dict = {}
	res_dict["lv1_labels"] = list(lv1_label_encoder.classes_)
	res_dict["lv2_labels"] = list(lv2_label_encoder.classes_)
	# model
	for lv1_dr, lv1_cls, lv2_dr, lv2_cls in itertools.product(
		args.level1_dim_reduc, args.level1_classifier,
		args.level2_dim_reduc, args.level2_classifier):
		# log running model
		model_name = "lv1-%s-%s.lv2-%s-%s" % (lv1_dr, lv1_cls, lv2_dr, lv2_cls)
		res_dict["model"] = model_name
		# create model
		print("model:", model_name)
		try:
			# properties of level setups
			level1_props = _make_level_props_dict(lv1_dr, lv1_cls,\
				args.level1_reduce_dim_to)
			level2_props = _make_level_props_dict(lv2_dr, lv2_cls,\
				args.level2_reduce_dim_to)
			#print(level2_props)
			model = pylib.TwoLevelModel(\
				level1_props = level1_props,\
				level2_props = level2_props,\
				indep_level2 = args.indep_level2)
			# cross validation
			cv = pylib.TwoLevelCrossValidator(model, args.cv_folds, args.permutation)
			# run cv
			cv.run_cv(data, encoded_univ_labels, encoded_lv1_labels, encoded_lv2_labels)
			# results output
			res_dict["evaluation"] = cv.evaluation.copy()
		except Exception as e:
			res_dict["evaluation"] = "error:" + repr(e)
		########################################################################
		# output
		# for task-level parallelism, we write output into separate files
		txt_ofile = ".".join([\
			os.path.basename(args.config),\
			model_name,\
			"%d_fold" % args.cv_folds,\
			"lv2_indep" if args.indep_level2 else "lv2_dep",\
			"json"])
		txt_ofile = os.path.join(args.output_txt_dir, txt_ofile)
		#
		os.makedirs(args.output_txt_dir, exist_ok = True)
		with open(txt_ofile, "w") as fh:
			json.dump(res_dict, fh)
	return


def main():
	args = get_args()
	# check and create dataset
	if not pylib.DatasetCollection.check_query_subclass(args.dataset,
		exp_cls = pylib.dataset.DuoLabelDatasetBase):
		raise RuntimeError("dataset used by 2-level method must be "
			"duo-labelled (both phase and strain), '%s' is incompatible"\
			% args.dataset)
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
	return


if __name__ == "__main__":
	main()
