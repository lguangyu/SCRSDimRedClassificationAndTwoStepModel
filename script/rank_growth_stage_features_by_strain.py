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
		metavar = "model",
		choices = pylib.FeatureRankCollection.get_registered_keys(),
		help = "feature rank method (required); choices: %s"\
			% pylib.FeatureRankCollection.repr_reg_keys())
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "tsv",
		help = "write output to this file instead of stdout")

	args = ap.parse_args()
	# refine args
	if args.output == "-":
		args.output = sys.stdout
	return args


def rank_growth_stage_features_by_strain(dataset, method: str) -> dict:
	ret = dict()
	rank_meth = pylib.FeatureRankCollection.query(method)()
	for l in numpy.unique(dataset.strain_label):
		mask = dataset.strain_label == l
		rank = rank_meth.rank_features(
			X = dataset.data[mask, :],
			Y = dataset.phase_label[mask],
		)
		# add to return dict
		text_label = dataset.strain_label_encoder.inverse_transform([l])[0]
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
	rank_dict = rank_growth_stage_features_by_strain(dataset, args.rank_method)
	# output
	with pylib.util.file_io.get_fh(args.output) as fp:
		for k in sorted(rank_dict.keys()):
			print(("\t").join([k] + [str(i) for i in rank_dict[k].tolist()]),
				file = fp)
	return


if __name__ == "__main__":
	main()
