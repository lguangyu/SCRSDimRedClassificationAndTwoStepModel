#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot
import numpy
#import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "-d", "--dataset", type = str, required = True,
		metavar = "dataset",
		choices = pylib.DatasetCollection.get_registered_keys(),
		help = "the dataset to run model on (required); choices: %s"\
			% pylib.DatasetCollection.repr_reg_keys())
	ap.add_argument("-P", "--prefix", type = str, default = "",
		metavar = "prefix",
		help = "prefix of output image files (default: <empty>)")
	ap.add_argument("-l", "--labels", type = str, default = None,
		metavar = "str",
		help = "only draw spectra from this given label names; multiple names "
			"should be comma-delimited; if not given (as default), draw all")
	ap.add_argument("--draw-label-stats", action = "store_true",
		help = "draw label stats rather than each single spectrum individually "
			"(default: no)")

	args = ap.parse_args()
	# refine args
	return args


def setup_layout(figure):
	layout = dict(figure = figure)

	# figure margins
	left_margin_inch	= 0.7
	right_margin_inch	= 0.3
	top_margin_inch		= 0.3
	bottom_margin_inch	= 0.7

	# spec axes dimensions
	spec_width_inch		= 6.0
	spec_height_inch	= 3.0

	# figure dimensions
	fig_width_inch	= left_margin_inch + spec_width_inch + right_margin_inch
	fig_height_inch	= bottom_margin_inch + spec_height_inch + top_margin_inch
	figure.set_size_inches(fig_width_inch, fig_height_inch)

	# create axes
	spec_left	= left_margin_inch / fig_width_inch
	spec_bottom	= bottom_margin_inch / fig_height_inch
	spec_width	= spec_width_inch / fig_width_inch
	spec_height	= spec_height_inch / fig_height_inch
	spec = figure.add_axes([spec_left, spec_bottom, spec_width,
		spec_height])
	layout["spec"] = spec

	# apply axes style
	for sp in spec.spines.values():
		sp.set_visible(False)
	spec.tick_params(
		left = False, labelleft = False,
		right = False, labelright = False,
		bottom = False, labelbottom = False,
		top = False, labeltop = False)

	return layout


def draw(png_prefix, data, *, as_stats: True, title = "", dpi = 300):
	color	= "#6060f0"
	x		= numpy.arange(len(data[0]))
	xlim	= (x.min(), x.max())

	if as_stats:
		# create layout
		layout	= setup_layout(matplotlib.pyplot.figure())
		figure	= layout["figure"]
		axes	= layout["spec"]
		# get y stats
		yavg = data.mean(axis = 0)
		yerr = data.std(axis = 0)
		ymin = data.min(axis = 0)
		ymax = data.max(axis = 0)
		# plot
		axes.plot(x, yavg, linestyle = "-", linewidth = 1.0, color = color)
		axes.fill_between(x, yavg - yerr, yavg + yerr, edgecolor = "none",
			facecolor = color + "60")
		axes.fill_between(x, ymin, ymax, edgecolor = "#ffffff00",
			facecolor = color + "20")
		# misc
		axes.set_xlim(*xlim)
		axes.set_title(title)
		# save and clean up
		matplotlib.pyplot.savefig(png_prefix + ".png", dpi = dpi)
		matplotlib.pyplot.close()
	else:
		# plot each spectra
		for i, y in enumerate(data):
			# create layout
			layout	= setup_layout(matplotlib.pyplot.figure())
			figure	= layout["figure"]
			axes	= layout["spec"]
			# plot
			axes.plot(x, yavg, linestyle = "-", linewidth = 1.0, color = color)
			# misc
			sfx = "-%03u" % (i + 1)
			axes.set_xlim(*xlim)
			axes.set_title(title + sfx)
			# save and clean up
			matplotlib.pylib.savefig(png_prefix + sfx + ".png", dpi = dpi)
			matplotlib.pylib.close()
	return


def draw_dataset_labels(prefix, dataset, labels, *, as_stats = True, dpi = 300):
	# get labels
	labels = labels.split(",") if labels is not None\
		else numpy.unique(dataset.text_label)
	# draw
	for l in labels:
		data = dataset.data[dataset.text_label == l]
		draw(prefix + l, data, as_stats = as_stats, title = l, dpi = dpi)
	return


def main():
	args = get_args()
	# check and create dataset
	if not pylib.DatasetCollection.check_query_subclass(args.dataset,
		exp_cls = pylib.dataset.SingleLabelDatasetBase):
		raise RuntimeError("input dataset must be single-labelled (only strain "
			"or phase), '%s' is incompatible" % args.dataset)
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
	# draw labels
	draw_dataset_labels(args.prefix, dataset, labels = args.labels,
		as_stats = args.draw_label_stats)
	return


if __name__ == "__main__":
	main()
