#!/usr/bin/env python3

import sys
import os
import numpy
import matplotlib
import matplotlib.lines
import matplotlib.pyplot
import argparse


DIMSTEST_DATASETS = [
	"COMBINED.phase",
	"COMBINED.strain",
	"EXPONENT1-50",
	"PLATFORM1-50",
	"PLATFORM2-50",
]

DIMSTEST_CLASSIFIERS = [
	dict(id = "gnb", display_name = "GNB"),
	dict(id = "lr", display_name = "LR"),
	dict(id = "lda", display_name = "LDA"),
	dict(id = "svm_lin", display_name = "Linear SVM"),
	dict(id = "svm_rbf", display_name = "RBF SVM"),
]

DIMSTEST_DIMREDUC = [
	dict(id = "pca",		display_name = "PCA"),
	dict(id = "lda",		display_name = "LDA"),
	dict(id = "lsdr",	display_name = "LSDR"),
]

DIMSTEST_DIMS = numpy.arange(30) + 1


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input_dir", type = str,
		help = "input directory")
	ap.add_argument("-o", "--output-dir", type = str,
		metavar = "dir", default = "./image",
		help = "output directory (default: ./image)")
	args = ap.parse_args()
	return args


def load_data_from_log(fname):
	with open(fname, "r") as fh:
		lines = fh.read().splitlines()
		n_lines = len(lines)
	block_size = 27
	res = []
	for block in even_chunks(lines, block_size):
		res.append(parse_block(block))
	train, test = _transform_dicts(res)
	return train, test


def even_chunks(iterable, size):
	for i in range(0, len(iterable), size):
		yield iterable[i:i + size]


def parse_block(lines):
	if len(lines) != 27:
		raise RuntimeError("corrupted block")
	ret = dict(train = [], test = [])
	for l in lines[-20:]:
		s = l.split(" ")
		v = float(s[-1])
		if s[0] == "training":
			ret["train"].append(v)
		elif s[0] == "test":
			ret["test"].append(v)
		else:
			raise RuntimeError("corrupted block")
	return ret


def _transform_dicts(_list):
	# _list[#1-30]{"train","test"}
	train = list(map(lambda i: i["train"], _list))
	train = numpy.asarray(train, dtype = float)
	test = list(map(lambda i: i["test"], _list))
	test = numpy.asarray(test, dtype = float)
	return train, test


def plot_log(axes, log_file):
	#
	train, test = load_data_from_log(log_file)
	x = numpy.arange(len(train)) + 1
	train_mean = train.mean(axis = 1)
	train_std = train.std(axis = 1)
	test_mean = test.mean(axis = 1)
	test_std = test.std(axis = 1)
	#
	axes.plot(x, train_mean, clip_on = False,
		linestyle = "-", linewidth = 1.0,
		color = "#4040FF", label = "Train accuracy")
	axes.plot(x, test_mean, clip_on = False,
		linestyle = "-", linewidth = 1.0,
		color = "#FF8000", label = "Test accuracy")
	return


def plot_all(dataset_res_dir, png):
	figure, axes_list = matplotlib.pyplot.subplots(
		nrows = len(DIMSTEST_CLASSIFIERS), ncols = len(DIMSTEST_DIMREDUC),
		figsize = (6.5, 9), sharex = True, sharey = True)
	matplotlib.pyplot.subplots_adjust(
		left = 0.15, right = 0.95, bottom = 0.11, top = 0.96,
		hspace = 0.04, wspace = 0.04)
	figure.align_ylabels(axes_list[:, 0])
	#
	_legend_set = False
	#
	for ax_c, dr in enumerate(DIMSTEST_DIMREDUC):
		for ax_r, classifier in enumerate(DIMSTEST_CLASSIFIERS):
			ax = axes_list[ax_r, ax_c]
			# appearance
			ax.set_facecolor("#F0F0F0")
			for sp in ax.spines.values():
				sp.set_visible(False)
			ax.grid(linestyle = "-", linewidth = 1.0, color = "#FFFFFF")
			# column heads
			if ax_r == 0:
				ax.text(x = 15, y = 1.05, s = dr["display_name"],
					clip_on = False, fontsize = 14, color = "#4040FF",
					horizontalalignment = "center", verticalalignment = "bottom")
			# row labels
			if ax_c == 0:
				ax.set_ylabel(classifier["display_name"], fontsize = 14,
					color = "#4040FF")
			# x labels
			if ax_r == (len(axes_list) - 1):
				ax.set_xlabel("Dimensions", fontsize = 14)
			# hide ticks
			tick_showleft = (ax_c == 0)
			tick_showbottom = (ax_r == (len(axes_list) - 1))
			ax.tick_params(
				left = tick_showleft, labelleft = tick_showleft,
				right = False, labelright = False,
				bottom = tick_showbottom, labelbottom = tick_showbottom,
				top = False, labeltop = False)
			ax.set_xlim(1.0, 30.0)
			ax.set_ylim(0.0, 1.0)
			ax.set_xticks([10, 20, 30])
			ax.set_yticks([0.25, 0.50, 0.75, 1.00])
			# plot
			fdata = os.path.join(dataset_res_dir, dr["id"], classifier["id"], "log")
			try:
				plot_log(ax, fdata)
			except:
				pass
	# legend, only once
	figs = [
		matplotlib.pyplot.Line2D([], [], linewidth = 1.0, linestyle = "-",
			color = "#4040FF", label = "Train accuracy"),
		matplotlib.pyplot.Line2D([], [], linewidth = 1.0, linestyle = "-",
			color = "#FF8000", label = "Test accuracy"),
	]
	figure.legend(handles = figs, ncol = 2, loc = "lower center")
	matplotlib.pyplot.savefig(png)
	matplotlib.pyplot.close()


if __name__ == "__main__":
	args = get_args()
	for dataset in DIMSTEST_DATASETS:
		res_dir = os.path.join(args.input_dir, dataset)
		png = os.path.join(args.output_dir, "%s.dims_test.png" % dataset)
		plot_all(res_dir, png)
