#!/usr/bin/env python3

import argparse
import itertools
import matplotlib
import matplotlib.pyplot
import numpy
import sys
# custom lib
import pylib


class CommaSeparatedList(list):
	def __init__(self, s, *ka, **kw):
		_list = s.split(",") if isinstance(s, str) else s
		return super().__init__(_list, *ka, **kw)


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--datasets", type = CommaSeparatedList,
		metavar = "ds1[,ds2[,...]]",
		default = "exponential,platform-1,platform-2",
		help = "list of dataset to analyze, separated by comma (,);"
			"(default: exponential,platform-1,platform-2); possible choices: "\
			+ (", ".join(pylib.DatasetCollection.get_registered_keys()))\
			+ "; **some entries may have multiple aliases")
	ap.add_argument("-p", "-o", "--plot", type = str, default = "-",
		metavar = "png",
		help = "save plot png to this file instead of stdout")
	args = ap.parse_args()
	# refine args
	if args.plot == "-":
		args.plot = sys.stdout.buffer
	return args


def _setup_layout(n_datasets) -> dict:
	# margins
	left_margin_inch	= 0.2
	right_margin_inch	= 0.2
	bottom_margin_inch	= 0.2
	top_margin_inch		= 0.2
	# dimensions of axes
	ylabel_pad_inch		= 0.5
	xlabel_pad_inch		= 0.8
	axes_width_inch		= 6.0
	axes_height_inch	= 1.8
	axes_hgap_inch		= 0.1
	# dimensions of entire grid
	axes_grid_width_inch	= axes_width_inch
	axes_grid_height_inch	= n_datasets * axes_height_inch\
		+ (n_datasets - 1) * axes_hgap_inch
	axes_grid_left_inch		= left_margin_inch + ylabel_pad_inch
	axes_grid_bottom_inch	= bottom_margin_inch + xlabel_pad_inch
	# total text in axes
	total_text_offsetx_inch	= -0.1
	total_text_offsety_inch	= -0.1

	# create figure
	figure_width_inch	= left_margin_inch + ylabel_pad_inch\
		+ axes_grid_width_inch + right_margin_inch
	figure_height_inch	= bottom_margin_inch + xlabel_pad_inch\
		+ axes_grid_height_inch + top_margin_inch
	figure = matplotlib.pyplot.figure(
		figsize = (figure_width_inch, figure_height_inch))

	# create subplots
	axes_list = figure.subplots(nrows = n_datasets, sharex = True, sharey = True,
		squeeze = False, gridspec_kw = dict(
			left	= axes_grid_left_inch / figure_width_inch,
			bottom	= axes_grid_bottom_inch / figure_height_inch,
			right	= (axes_grid_left_inch + axes_grid_width_inch)\
				/ figure_width_inch,
			top		= (axes_grid_bottom_inch + axes_grid_height_inch)\
				/ figure_height_inch,
			hspace	= axes_hgap_inch / axes_height_inch
		)
	)

	# return dict
	ret = {
		"figure": figure,
		"axes_list": axes_list.squeeze(axis = 1),
		"total_text_topright": {
			"x": 1.0 + total_text_offsetx_inch / axes_width_inch,
			"y": 1.0 + total_text_offsety_inch / axes_height_inch,
		},
	}
	return ret


def _count_dataset_labels(dataset_name) -> dict:
	ds = pylib.DatasetCollection.get_dataset(dataset_name)
	if not isinstance(ds, pylib.dataset.SingleLabelDataset):
		raise ValueError("dataset must be SingleLabelDataset, but the specified"
			" '%s' is not" % dataset_name)
	label, counts = numpy.unique(ds.text_label, return_counts = True)
	return dict(zip(label, counts))


def plot_dataset_class_histogram(plot_file, dataset_names):
	n_datasets = len(dataset_names)
	ds_lab_counts = [_count_dataset_labels(i) for i in dataset_names]
	# get a sorted list of all labels
	uniq_label = itertools.chain(*map(lambda i: i.keys(), ds_lab_counts))
	uniq_label = sorted(set(uniq_label))
	n_uniq_label = len(uniq_label)

	# setup layout
	layout = _setup_layout(n_datasets)

	# plot each dataset onto a different axes
	x = numpy.arange(n_uniq_label) + 0.5
	for axes, cnts, name in\
		zip(layout["axes_list"], ds_lab_counts, dataset_names):
		# style axes
		for sp in axes.spines.values():
			sp.set_visible(False)
		axes.set_facecolor("#E8E8F0")
		axes.grid(linestyle = "-", linewidth = 1.0, color = "#FFFFFF",
			axis = "y", zorder = 1)
		# get counts data (y)
		# if label not present in that dataset, use 0
		y = [cnts.get(i, 0) for i in uniq_label]
		assert len(y) == n_uniq_label, len(y)
		# bar plot
		axes.bar(x, y, width = 1, align = "center", facecolor = "#63709480",
			fill = True, zorder = 2)
		# add total text
		axes.text(**layout["total_text_topright"],
			s = "Total samples = %d" % sum(cnts.values()),
			fontsize = 12, fontweight = "semibold", transform = axes.transAxes,
			horizontalalignment = "right", verticalalignment = "top")
		# misc
		axes.set_xlim(0, n_uniq_label)
		axes.set_xticks(x)
		axes.set_xticklabels(uniq_label, rotation = 90)
		axes.set_ylim(0, 100)
		axes.set_ylabel(name.upper())

	# save figure
	matplotlib.pyplot.savefig(plot_file, dpi = 300)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	plot_dataset_class_histogram(args.plot, args.datasets)
	return


if __name__ == "__main__":
	main()
