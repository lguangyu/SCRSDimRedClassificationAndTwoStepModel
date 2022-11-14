#!/usr/bin/env python3

import argparse
import numpy
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
	ap.add_argument("-m", "--rank-method", type = str, required = True,
		metavar = "str",
		choices = pylib.FeatureRankCollection.get_registered_keys(),
		help = "feature rank method (required); choices: %s"\
			% pylib.FeatureRankCollection.repr_reg_keys())
	#ap.add_argument("-l", "--strain-list", type = str, default = None,
	#	metavar = "str[,str...]",
	#	help = "run feature rank on a list of selected strains; multiple "
	#		"strains are separated by comma (,)i; if remain unspecified, will "
	#		"run on all strains found in the given dataset")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "tsv",
		help = "write output to this file instead of stdout")

	args = ap.parse_args()
	# refine args
	if args.output == "-":
		args.output = sys.stdout
	return args


def rank_feature_strain_relevance_by_growth_stage(dataset, method: str,
		phase_list = None) -> dict:
	ret = dict()
	if phase_list is None:
		phase_list = numpy.unique(dataset.phase_text_label)
	else:
		phase_list = phase_list.split(",")
	# calcualte for each strain label
	for text_label in phase_list:
		label = dataset.phase_label_encoder.transform([text_label])[0]
		mask = dataset.phase_label == label
		rank_meth = pylib.FeatureRankCollection.query(method)()
		rank = rank_meth.rank_features(
			X = dataset.data[mask, :],
			Y = dataset.strain_label[mask],
		)
		# add to return dict
		ret[text_label] = rank
	return ret


def main():
	args = get_args()
	# check and create dataset
	if not pylib.DatasetCollection.check_query_subclass(args.dataset,
		exp_cls = pylib.dataset.DuoLabelDatasetBase):
		raise RuntimeError("dataset used by 2-level method must be "
			"duo-labelled (both phase and strain), '%s' is incompatible"\
			% args.dataset)
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
	# calculate feature rank
	rank_dict = rank_feature_strain_relevance_by_growth_stage(
		dataset, args.rank_method)
		#, phase_list = args.strain_list)
	# output
	with pylib.util.file_io.get_fh(args.output, "w") as fp:
		for k in sorted(rank_dict.keys()):
			print(("\t").join([k] + [str(i) for i in rank_dict[k].tolist()]),
				file = fp)
	return


if __name__ == "__main__":
	main()
