#!/usr/bin/env python3

import numpy
import os
import sys
import argparse


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--normalize", type = str,
		choices = ["none", "l1", "l2", "minmax", "area"],
		default = "l2", help = "normalization method")
	return ap.parse_args()


def parse_file(fname, normalize):
	with open(fname, "r") as fh:
		d = fh.read().splitlines()
	d = [i.split("\t")[1] for i in d]
	d = numpy.asarray(d, dtype = float)
	if normalize == "none":
		pass
	elif normalize == "l1":
		d = d / numpy.abs(d).sum()
	elif normalize == "l2":
		scale = numpy.sqrt(numpy.dot(d, d))
		d = d / scale
	elif normalize == "minmax":
		dmin, dmax = d.min(), d.max()
		scale = dmax - dmin
		d = (d - dmin) / scale
	elif normalize == "area":
		d = d / d.sum()
	else:
		raise ValueError("unrecognized normalization method")
	return d


def combine_dataset(path, name, normalize):
	# combine a dataset to a single file
	fh_meta = open("./data/%s.normalized_%s.meta.tsv" % (name, normalize), "w")
	fh_data = open("./data/%s.normalized_%s.data.tsv" % (name, normalize), "w")
	print("%s" % name, file = sys.stderr)
	for s in os.scandir(path):
		if s.is_dir():
			print("%s" % s.name, file = sys.stderr)
			for f in os.scandir(s.path):
				if f.is_file():
					#print("processing: %s" % f.path, file = sys.stderr)
					# label, file
					print("\t".join([s.name, f.name]), file = fh_meta)
					data = parse_file(f.path, normalize)
					print("\t".join(["%f" % i for i in data]), file = fh_data)
	fh_meta.close()
	fh_data.close()
	return


if __name__ == "__main__":
	args = get_args()
	os.makedirs("./data", exist_ok = True)
	for i in os.scandir("./raw"):
		combine_dataset(i.path, i.name, args.normalize)
