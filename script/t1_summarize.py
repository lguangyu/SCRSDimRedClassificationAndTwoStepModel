#!/usr/bin/env python3

import argparse
import itertools
import json
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
import numpy
import os
import sys
import warnings


SUMMARIZE_CLASSIFIERS = [
	dict(id = "gnb", display_name = "GNB"),
	dict(id = "lr", display_name = "LR"),
	dict(id = "lda", display_name = "LDA"),
	dict(id = "svm_lin", display_name = "Linear SVM"),
	dict(id = "svm_rbf", display_name = "RBF SVM"),
	dict(id = "svm_lin_cv", display_name = "Lin-SVM (CV)"),
	dict(id = "svm_rbf_cv", display_name = "KSVM (CV)"),
]

SUMMARIZE_DIMREDUC = [
	dict(id = "none_none",	display_name = "N/A"),
	dict(id = "pca_26",		display_name = "PCA(26)"),
	dict(id = "lda_26",		display_name = "LDA(26)"),
	dict(id = "lsdr_26",	display_name = "LSDR(26)"),
]


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-prefix", type = str,
		metavar = "prefix", required = True,
		help = "prefix of cross validation results (required)")
	ap.add_argument("-o", "--output-prefix", type = str,
		metavar = "prefix", required = True,
		help = "prefix of output summary (required)")
	ap.add_argument("-d", "--delimiter", type = str,
		metavar = "char", default = "\t",
		help = "delimiter in input/output file (default: <tab>)")
	ap.add_argument("-m", "--metric", type = str,
		metavar = "str", default = "average_accuracy",
		help = "metric to of model performance evaluation "
			"(default: average_accuracy)")
	args = ap.parse_args()
	return args


def get_results_from_json(fname, eval_key) -> list:
	if not os.path.isfile(fname):
		warnings.warn("file '%s' does not exists, skipping" % fname)
		return None
	with open(fname, "r") as fh:
		res = json.load(fh)
		# extract testing average_accuracy from all folds
		ret = [i["testing"]["average_accuracy"] for i in res["folds"]]
	return ret


def load_and_combine_results(prefix, eval_key):
	# load/extract results <rowtag> from multiple files
	# organize 1st dim as classification method
	# 2nd dim as dim reduction methods
	# 3nd dim as folds from cross validation
	ret = []
	fname_proto = "%s.%s.dr_%s.10_fold.txt"
	for cls in SUMMARIZE_CLASSIFIERS:
		r = []
		for dr in SUMMARIZE_DIMREDUC:
			ifn = fname_proto % (prefix, cls["id"], dr["id"])
			_res = get_results_from_json(ifn, eval_key)
			r.append([numpy.nan] * 10 if _res is None else _res)
		ret.append(r)
	ret = numpy.asarray(ret, dtype = float)
	assert ret.ndim == 3
	return ret


def compute_mean_stdev(data):
	"""
	compute mean and stdev of a 3-d array along its third dimension
	some cell may appear a None
	"""
	mean = data.mean(axis = 2)
	stdev = data.std(axis = 2)
	return mean, stdev


def savetxt(fname, mean, stdev, delimiter):
	with open(fname, "w") as fh:
		# header
		col_headers = list(map(lambda i: i["display_name"], SUMMARIZE_DIMREDUC))
		header = delimiter.join([""] + col_headers)
		print(header, file = fh)
		# data
		for i, classifier in enumerate(SUMMARIZE_CLASSIFIERS):
			# each line is a classifier
			line = [classifier["display_name"]] # classifier name
			for j, dimreduc in enumerate(SUMMARIZE_DIMREDUC):
				_str = "%.3f (%.2f)" % (mean[i, j], stdev[i, j])
				line.append(_str)
			print(delimiter.join(line), file = fh)
	return


def save_tableplot(fname, mean, stdev, title = ""):
	assert mean.shape == stdev.shape
	# layout
	nrow, ncol			= mean.shape
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
	table = table_axes.pcolor(mean, cmap = cmap, vmin = 0.0, vmax = 1.0,
		alpha = table_patch_alpha)
	# add text to cell
	for r, c in itertools.product(range(nrow), range(ncol)):
		_vmean, _vstd = mean[r, c], stdev[r, c]
		text = "N/A" if numpy.isnan(_vmean) else ("%.3f (%.2f)" % (_vmean, _vstd))
		# if blackground is dark, use white; else black
		color = "#FFFFFF" if _vmean > 0.6 else "#000000"
		table_axes.text(x = c + 0.5, y = r + 0.5, s = text, color = color,
			fontsize = 12, fontweight = "bold",
			horizontalalignment = "center", verticalalignment = "center")

	# misc
	# x labels
	xticks = numpy.arange(ncol) + 0.5
	xticklabels = [i["display_name"] for i in SUMMARIZE_DIMREDUC]
	table_axes.set_xticks(xticks)
	table_axes.set_xticklabels(xticklabels, color = "#000000", fontsize = 14,
		horizontalalignment = "center", verticalalignment = "bottom")
	# y labels
	yticks = numpy.arange(nrow) + 0.5
	yticklabels = [i["display_name"] for i in SUMMARIZE_CLASSIFIERS]
	table_axes.set_yticks(yticks)
	table_axes.set_yticklabels(yticklabels, color = "#000000", fontsize = 14,
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
	## yticks
	#cmax_ymin, cmax_ymax = 128, 256
	#cmax_yrange = cmax_ymax - cmax_ymin
	#yticks = numpy.linspace(cmax_ymin, cmax_ymax, 5)
	#yticklabels = ["%.1f" % ((i - cmax_ymin) / cmax_yrange) for i in yticks]
	#cmax.set_ylim(cmax_ymin, cmax_ymax)
	#cmax.set_yticks(yticks)
	#cmax.set_yticklabels(yticklabels)
	## misc
	figure.suptitle(title, fontsize = 18,
		horizontalalignment = "center", verticalalignment = "top")
	matplotlib.pyplot.savefig(fname, dpi = 300)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	data = load_and_combine_results(args.input_prefix, args.metric)
	# calculate mean and std
	mean, std = compute_mean_stdev(data)
	# output
	os.makedirs(os.path.dirname(args.output_prefix), exist_ok = True)
	#savetxt(args.output_prefix + ".tsv", mean, std, args.delimiter)
	# plot
	title = "%s, %s summary" % (args.input_prefix, args.metric)
	fname = "%s.%s.summary.png" % (args.output_prefix, args.metric)
	save_tableplot(fname, mean, std, title = title)
	return


if __name__ == "__main__":
	main()
