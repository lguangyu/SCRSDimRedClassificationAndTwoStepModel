#!/usr/bin/env python3

import numpy
import os
import sys
from matplotlib import pyplot


datasets = [
	dict(file = "./data/EXPONENT1-50.normalized_l2.meta.tsv", name = "EXPONENTIAL"),
	dict(file = "./data/PLATFORM1-50.normalized_l2.meta.tsv", name = "PLATFORM1"),
	dict(file = "./data/PLATFORM2-50.normalized_l2.meta.tsv", name = "PLATFORM2"),
]


def get_labels(fname):
	with open(fname, "r") as fh:
		lines = fh.read().splitlines()
	labels = [i.split("\t")[0] for i in lines]
	return numpy.asarray(labels, dtype = object)


def count_labels(labels):
	uniq, count = numpy.unique(labels, return_counts = True)
	assert len(uniq) == len(count)
	return dict(zip(uniq, count))


if __name__ == "__main__":
	for d in datasets:
		labels = get_labels(d["file"])
		d["total"] = len(labels)
		d["labels"] = count_labels(labels)
	all_labels = numpy.concatenate([list(d["labels"].keys()) for d in datasets])
	all_labels = numpy.unique(all_labels.astype(object))
	all_labels.sort()

	fig, axes = pyplot.subplots(nrows = len(datasets), figsize = (6, 6), sharex = True)
	pyplot.subplots_adjust(left = 0.10, right = 0.95, bottom = 0.15, top = 0.95,
		hspace = 0.10)

	for d, ax in zip(datasets, axes):
		x = numpy.arange(len(all_labels)) + 0.5
		c = [d["labels"].get(l, 0) for l in all_labels]
		ax.bar(x, c, width = 1, align = "center", fill = True,
			facecolor = "#FF8000")
		ax.text(0.98, 0.94, s = "Total=%d" % d["total"], fontsize = 14,
			horizontalalignment = "right", verticalalignment = "top",
			transform = ax.transAxes)
		ax.set_ylabel(d["name"])
		ax.set_ylim(0, 100)

	ax = axes[-1]
	ax.set_xlim(0, len(all_labels))
	ax.set_xticks(x)
	ax.set_xticklabels(all_labels, rotation = 270)

	pyplot.savefig("image/hist_labels.png")
	pyplot.close()
