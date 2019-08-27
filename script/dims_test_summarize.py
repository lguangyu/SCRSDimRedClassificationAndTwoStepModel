#!/usr/bin/env python3

import argparse
#import collections
import glob
#import itertools
import json
import matplotlib
import matplotlib.lines
import matplotlib.pyplot
import numpy
import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", nargs = "+",
		help = "input dimension reduction scan results, wildcard accepted")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "json",
		help = "write summary to this json file instead of stdout")
	ap.add_argument("-m", "--metric", type = str,
		metavar = "str", default = "average_accuracy",
		help = "metric of model evaluation (default: average_accuracy)")
	ap.add_argument("-p", "--plot", type = str,
		metavar = "png",
		help = "also draw a plot (default: off)")
	args = ap.parse_args()
	# refine args
	if args.output == "-":
		args.output = sys.stdout
	return args


def load_all_results(inputs) -> list:
	assert isinstance(inputs, list)
	ret = list()
	for i in inputs:
		for f in glob.glob(i):
			with open(f, "r") as fp:
				ret.append(json.load(fp))
	return ret


class OrganizedResultDict(dict):
	def add_result(self, res):
		if not isinstance(res, pylib.result_parsing.SingleLevelJSONCondenser):
			raise TypeError("res must be SingleLevelJSONCondenser, not '%s'"\
				% type(res).__name__)
		self._update_nested_dict(
			# keys to descend
			res.dataset,
			res.dimreducer,
			res.classifier,
			"traininig",
			res.n_components,
			value = res.train_mean)
		self._update_nested_dict(
			# keys to descend
			res.dataset,
			res.dimreducer,
			res.classifier,
			"testing",
			res.n_components,
			value = res.test_mean)
		return

	def _update_nested_dict(self, *keys, value):
		if not keys:
			raise ValueError("keys cannot be empty")
		d = self
		keys = list(keys)
		keys.reverse()
		while keys:
			# descend into nested dict structure by a list of keys
			k = keys.pop()
			if keys: # if remaining keys after popping, descend
				# i.e. k is not the last key
				d = d.setdefault(k, dict())
			else:
				# if k is the last key, update
				d[k] = value
		return


def organize_results(results, metric) -> dict:
	_ds = set() # datasets
	_dr = set() # dimreducers
	_cf = set() # classifiers
	# res_dict is in root[dataset][dimred][classif][train/test][n_cmpnt] order
	res_dict = OrganizedResultDict()
	for i in results:
		res = pylib.result_parsing.SingleLevelJSONCondenser.parse(i, metric)
		_ds.add(res.dataset)
		_dr.add(res.dimreducer)
		_cf.add(res.classifier)
		res_dict.add_result(res)
	# 'none' should always the first of drs
	if "none" in _dr:
		_dr.remove("none")
		sorted_dr = ["none"] + sorted(_dr)
	return dict(datasets = sorted(_ds), dimreducers = sorted_dr,
		classifiers = sorted(_cf), results = res_dict)


# TODO: finish the plots
class Plot(object):
	pass


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






def main():
	args = get_args()
	results = organize_results(load_all_results(args.input), args.metric)
	# output
	with pylib.util.file_io.get_fh(args.output, "w") as fp:
		json.dump(results, fp)
	return


if __name__ == "__main__":
	main()
