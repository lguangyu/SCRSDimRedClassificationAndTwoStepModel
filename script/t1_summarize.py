#!/usr/bin/env python3

import argparse
import glob
import itertools
import json
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
import numpy
import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", nargs = "+",
		help = "input cross validation results, wildcard accepted")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "tsv",
		help = "write summary table to this file instead of stdout")
	ap.add_argument("-m", "--metric", type = str,
		metavar = "str", default = "average_accuracy",
		help = "metric of model evaluation (default: average_accuracy)")
	ap.add_argument("-p", "--plot", type = str,
		metavar = "png",
		help = "also output a tabular plot (default: off)")
	args = ap.parse_args()
	# refine args
	if args.output == "-":
		args.output = sys.stdout
	return args


def load_all_results(inputs) -> list:
	assert isinstance(inputs, list)
	ret = []
	for i in inputs:
		for f in glob.glob(i):
			with open(f, "r") as fp:
				ret.append(json.load(fp))
	return ret


def condense_results(all_res, metric):
	assert isinstance(all_res, list), type(all_res)
	# dataset info
	dataset = pylib.result_parsing.as_unique(lambda x: x["dataset"], all_res)
	models = [pylib.result_parsing.SingleLevelJSONCondenser.parse(i, metric)\
		for i in all_res]
	# combine models
	drs = sorted(set(map(lambda x: x.dimreducer, models)))
	cfs = sorted(set(map(lambda x: x.classifier, models)))
	# rearange, ensure 'none' is the first in drs
	if "none" in drs:
		drs.remove("none")
		drs.insert(0, "none")
	# dimreducer as col, classifier as row
	nrow, ncol = len(cfs), len(drs)
	mean_mat = numpy.full((nrow, ncol), numpy.nan)
	std_mat = numpy.full((nrow, ncol), numpy.nan)
	for mdl, res in zip(models, all_res):
		rid = cfs.index(mdl.classifier)
		cid = drs.index(mdl.dimreducer)
		mean_mat[rid, cid]	= mdl.test_mean
		std_mat[rid, cid]	= mdl.test_std
	return dict(dataset = dataset,
		dimreducer_labels = drs,
		classifier_labels = cfs,
		mean_matrix = mean_mat,
		std_matrix = std_mat)


def save_txt(fp, mean_mat, std_mat, drs, cfs, delimiter = "\t"):
	fp.write("\t".join([""] + ["%s\tstd" % i for i in drs]) + "\n")
	for i, cf in enumerate(cfs):
		fp.write("\t".join([cf] +\
			["%.3f\t%.3f" % (mean_mat[i, j], std_mat[i, j])\
				for j in range(len(drs))])\
			+ "\n")
	return


def save_plot(fn, mean_mat, std_mat, drs, cfs, title = ""):
	assert mean_mat.shape == std_mat.shape
	# layout
	nrow, ncol			= mean_mat.shape
	left_margin_inch	= 0.2
	right_margin_inch	= 0.2
	top_margin_inch		= 0.2
	bottom_margin_inch	= 0.2
	# table
	table_patch_alpha		= 0.8
	table_left_pad_inch		= 1.6
	table_top_pad_inch		= 0.8
	table_cell_width_inch	= 1.6
	table_cell_height_inch	= 0.6
	table_width_inch		= table_cell_width_inch * ncol
	table_height_inch		= table_cell_height_inch * nrow
	# colorbar
	table_colorbar_gap_inch	= 0.2
	colorbar_right_pad_inch	= 0.5
	colorbar_width_inch		= 0.6
	colorbar_height_inch	= table_height_inch # align to table
	# create figure
	figure_width_inch		= left_margin_inch + table_left_pad_inch\
		+ table_width_inch + table_colorbar_gap_inch + colorbar_width_inch\
		+ colorbar_right_pad_inch + right_margin_inch
	figure_height_inch		= bottom_margin_inch + table_height_inch\
		+ table_top_pad_inch + top_margin_inch
	figure = matplotlib.pyplot.figure(figsize =\
		(figure_width_inch, figure_height_inch))
	# create table axes
	table_left		= (left_margin_inch + table_left_pad_inch) / figure_width_inch
	table_bottom	= bottom_margin_inch / figure_height_inch
	table_width		= table_width_inch / figure_width_inch
	table_height		= table_height_inch / figure_height_inch
	table_axes		= figure.add_axes([table_left, table_bottom,
		table_width, table_height])
	# create colorbar axes
	colorbar_left	= (left_margin_inch + table_left_pad_inch + table_width_inch\
		+ table_colorbar_gap_inch) / figure_width_inch
	colorbar_bottom	= table_bottom # align to table
	colorbar_width	= colorbar_width_inch / figure_width_inch
	colorbar_height	= colorbar_height_inch / figure_height_inch
	colorbar_axes		= figure.add_axes([colorbar_left, colorbar_bottom,
		colorbar_width, colorbar_height])
	# style table axes
	for sp in table_axes.spines.values():
		sp.set_visible(False)
	table_axes.tick_params(left = False, right = False, bottom = False,
		top = False, labelleft = True, labelright = False, labeltop = True,
		labelbottom = False)
	# style colorbar axes
	for sp in colorbar_axes.spines.values():
		sp.set_visible(False)

	# plot table
	cmap = matplotlib.colors.LinearSegmentedColormap(None, N = 1024, gamma = 1.0,
		segmentdata = {
			"red": [
				(0.0,	1.0,	1.0),
				(1.0,	1.0,	1.0),
			],
			"green": [
				(0.0,	1.0,	1.0),
				(1.0,	0.0,	0.0),
			],
			"blue": [
				(0.0,	1.0,	1.0),
				(1.0,	0.0,	0.0),
			],
		})
	table = table_axes.pcolor(mean_mat, cmap = cmap, vmin = 0.0, vmax = 1.0,
		alpha = table_patch_alpha)
	# add text to cell
	for r, c in itertools.product(range(nrow), range(ncol)):
		_vmean, _vstd = mean_mat[r, c], std_mat[r, c]
		text = "N/A" if numpy.isnan(_vmean) else ("%.3f (%.2f)" % (_vmean, _vstd))
		# if blackground is dark, use white; else black
		color = "#FFFFFF" if _vmean > 0.6 else "#000000"
		table_axes.text(x = c + 0.5, y = r + 0.5, s = text, color = color,
			fontsize = 12, fontweight = "bold",
			horizontalalignment = "center", verticalalignment = "center")

	# misc
	# x labels
	xticks = numpy.arange(ncol) + 0.5
	table_axes.set_xticks(xticks)
	table_axes.set_xticklabels(drs, color = "#000000", fontsize = 14,
		horizontalalignment = "center", verticalalignment = "bottom")
	# y labels
	yticks = numpy.arange(nrow) + 0.5
	table_axes.set_yticks(yticks)
	table_axes.set_yticklabels(cfs, color = "#000000", fontsize = 14,
		horizontalalignment = "right", verticalalignment = "center")
	# axis limits
	table_axes.set_xlim(0, ncol)
	table_axes.set_ylim(0, nrow)
	table_axes.invert_yaxis()

	# color bar
	colorbar = figure.colorbar(table, cax = colorbar_axes,
		alpha = table_patch_alpha)
	# misc
	colorbar.outline.set_visible(False)
	figure.suptitle(title, fontsize = 18,
		horizontalalignment = "center", verticalalignment = "top")

	# save
	matplotlib.pyplot.savefig(fn, dpi = 300)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	all_res = load_all_results(args.input)
	# summarize results
	sum_res = condense_results(all_res, args.metric)
	# output table
	with pylib.util.file_io.get_fh(args.output, "w") as fp:
		save_txt(fp, delimiter = "\t",
			drs = sum_res["dimreducer_labels"],
			cfs = sum_res["classifier_labels"],
			mean_mat = sum_res["mean_matrix"],
			std_mat = sum_res["std_matrix"])
	# output plot
	if args.plot:
		save_plot(args.plot, title = sum_res["dataset"],
			drs = sum_res["dimreducer_labels"],
			cfs = sum_res["classifier_labels"],
			mean_mat = sum_res["mean_matrix"],
			std_mat = sum_res["std_matrix"])
	return


if __name__ == "__main__":
	main()
