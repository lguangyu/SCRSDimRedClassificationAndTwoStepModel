#!/usr/bin/env python3

import numpy
import os
import sys
import argparse
import sklearn.preprocessing


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-dir", type = str, metavar = "path",
		required = True,
		help = "input directory (required)")
	ap.add_argument("-o", "--output-dir", type = str, metavar = "path",
		required = True,
		help = "output directory (required)")
	ap.add_argument("-n", "--normalize", type = str,
		choices = ["none", "l1", "l2", "minmax", "area"],
		default = "l2", help = "normalization method")
	ap.add_argument("-s", "--scale", action = "store_true",
		help = "scale data to 0-mean and 1-stdev (default: off)")
	return ap.parse_args()


def load_file(fname, extract_column = 1):
	# not using numpy.loadtxt since contains unicode char (?)
	with open(fname, "r") as fh:
		lines = fh.read().splitlines()
	data = [l.split("\t")[extract_column] for l in lines]
	data = numpy.asarray(data, dtype = float)
	return data


def normalized(data, norm = "none"):
	"""
	returned row-wise normalized data
	norm: none, l1, l2, minmax, area
	none does nothing
	"""
	if norm == "none":
		return data
	elif norm in ["l1", "l2"]:
		# as literal
		return sklearn.preprocessing.normalize(data, norm = norm)
	elif norm == "minmax":
		# normalize to min = 0, max = 1
		row_min = numpy.min(data, axis = 1, keepdims = True)
		row_max = numpy.max(data, axis = 1, keepdims = True)
		row_range = row_min - row_max
		return (data - row_min) / row_range
	elif norm == "area":
		# note area is not identical to l1
		# it considers negative numbers
		return data / numpy.sum(data, axis = 1, keepdims = True)
	else:
		raise ValueError("unrecognized method '%s'" % norm)
	return


def combine_dataset(phase_dir, phase_name, norm, scale):
	meta = []
	data = []
	for strain in os.scandir(phase_dir):
		if strain.is_dir():
			for file in os.scandir(strain.path):
				if file.is_file():
					meta.append("\t".join([strain.name, file.name]))
					data.append(load_file(file.path, 1))
	data = numpy.asarray(data, dtype = float)
	data = normalized(data, norm)
	if scale:
		data = sklearn.preprocessing.scale(data)
	return meta, data


if __name__ == "__main__":
	args = get_args()
	os.makedirs(args.output_dir, exist_ok = True)
	for phase in os.scandir(args.input_dir):
		if phase.is_dir():
			# this end up with EXPONENT1, PLATFORM1, PLATFORM2 three dirs
			meta_basename = phase.name
			data_basename = "%s.normalized_%s" % (phase.name, args.normalize)
			if args.scale:
				data_basename += ".scaled"
			meta, data = combine_dataset(phase.path, phase.name, args.normalize, args.scale)
			# output
			with open(os.path.join(args.output_dir, meta_basename + ".labels.tsv"), "w") as fh:
				for i in meta:
					print(i, file = fh)
			numpy.savetxt(os.path.join(args.output_dir, data_basename + ".tsv"),\
				data, fmt = "%f", delimiter = "\t")
