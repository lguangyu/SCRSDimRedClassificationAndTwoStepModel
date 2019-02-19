#!/usr/bin/env python3
################################################################################
# this script is used to combine all three datasets used in stage 1
# into two combined datasets used in stage 2
################################################################################
# 1. data = (3 combined), labels = exponential, platform-1, platform-2
# 2. data = (3 combined), labels = strains
################################################################################

import sys


DATASETS = [
	{
		"phase": "EXPONENTIAL",
		"data": "./data/EXPONENT1-50.normalized_l2.data.tsv",
		"meta": "./data/EXPONENT1-50.normalized_l2.meta.tsv",
	},
	{
		"phase": "PLATFORM1",
		"data": "./data/PLATFORM1-50.normalized_l2.data.tsv",
		"meta": "./data/PLATFORM1-50.normalized_l2.meta.tsv",
	},
	{
		"phase": "PLATFORM2",
		"data": "./data/PLATFORM1-50.normalized_l2.data.tsv",
		"meta": "./data/PLATFORM2-50.normalized_l2.meta.tsv",
	},
]


def combined_data(ofn_data):
	with open(ofn_data, "w") as ofh:
		for ds in DATASETS:
			with open(ds["data"], "r") as ifh:
				for line in ifh:
					ofh.write(line)
	return


def combined_phase_meta(ofn_meta):
	with open(ofn_meta, "w") as ofh:
		for ds in DATASETS:
			with open(ds["meta"], "r") as ifh:
				for line in ifh:
					# replace strain name with phase name
					sline = [ds["phase"]] + line.split("\t")[1:]
					ofh.write(("\t").join(sline))
	return


def combined_strain_meta(ofn_meta):
	with open(ofn_meta, "w") as ofh:
		for ds in DATASETS:
			with open(ds["meta"], "r") as ifh:
				for line in ifh:
					ofh.write(line)
	return


if __name__ == "__main__":
	combined_data("./data/COMBINED.normalized_l2.data.tsv")
	combined_phase_meta("./data/COMBINED.phase.normalized_l2.meta.tsv")
	combined_strain_meta("./data/COMBINED.strain.normalized_l2.meta.tsv")
