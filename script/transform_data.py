#!/usr/bin/env python3

import numpy
import os
import sys
import argparse
import sklearn.preprocessing


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--normalize", type = str,
		choices = ["none", "l1", "l2", "minmax", "area"],
		default = "l2", help = "normalization method")
	#ap.add_argument("-s", "--scale", action = "store_true",
	#	help = "scale data to 0-mean and 1-stdev (default: off)")
	return ap.parse_args()


def load_file(fname, extract_column = 1):
	# not using numpy.loadtxt since contains unicode char (?)
	with open(fname, "r") as fh:
		lines = fh.read().splitlines()
	splitted = [l.split("\t") for l in lines]
	data = list(map(lambda i: i[extract_column], splitted))
	data = numpy.asarray(data_text, dtype = float)
	return data


def normalize(data, norm = "none"):
	"""
	returned normalized data
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
		_min, _max = data.min(), data.max()
		_range = _max - _min
		return (data - _min) / _range
	elif norm == "area":
		# note area is not identical to l1
		# it considers negative numbers
		return data / data.sum()
	else:
		raise ValueError("unrecognized method '%s'" % norm)
	return


def combine_dataset(path, name, normalize):
	# combine a dataset (multiple files) to a single file
	fh_meta = open("./data/%s.normalized_%s.meta.tsv" % (name, normalize), "w")
	fh_data = open("./data/%s.normalized_%s.data.tsv" % (name, normalize), "w")
	print("%s" % name, file = sys.stderr)
	for s in os.scandir(path):
		if s.is_dir():
			for f in os.scandir(s.path):
				if f.is_file():
					#print("processing: %s" % f.path, file = sys.stderr)
					# label, file in meta data
					print("\t".join([s.name, f.name]), file = fh_meta)
					data = parse_file(f.path, 1) # intensity is 2nd column (1)
					data = normalize(data, normalize)
					print("\t".join(["%f" % i for i in data]), file = fh_data)
	fh_meta.close()
	fh_data.close()
	return


if __name__ == "__main__":
	args = get_args()
	os.makedirs("./data", exist_ok = True)
	for i in os.scandir("./raw"):
		combine_dataset(i.path, i.name, args.normalize)
