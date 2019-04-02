#!/usr/bin/env python3

import sys
import os
import argparse
import numpy
import warnings


SUMMARIZE_CLASSIFIERS = [
	dict(id = "gnb", display_name = "GNB"),
	dict(id = "lr", display_name = "LR"),
	dict(id = "lda", display_name = "LDA"),
	dict(id = "svm_lin", display_name = "Linear SVM"),
	dict(id = "svm_rbf", display_name = "RBF SVM"),
]

SUMMARIZE_DIMREDUC = [
	dict(id = "none_none",	display_name = "N/A"),
	dict(id = "pca_%d",		display_name = "PCA(%d)"),
	dict(id = "lda_%d",		display_name = "LDA(%d)"),
	dict(id = "lsdr_%d",	display_name = "LSDR(%d)"),
]


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-prefix",
		type = str, metavar = "prefix", required = True,
		help = "prefix of cross validation results (required)")
	ap.add_argument("-o", "--output-prefix",
		type = str, metavar = "prefix", required = True,
		help = "prefix of output summary (required)")
	ap.add_argument("-d", "--delimiter",
		type = str, metavar = "char", default = "\t",
		help = "delimiter in input/output file (default: <tab>)")
	ap.add_argument("-r", "--reduc-dims-to",
		type = int, metavar = "int", required = True,
		help = "summary the results with this dimension reduction methods are\
			trained to reduce to this dimension (required)")
	ap.add_argument("-t", "--row-tag",
		type = str, metavar = "str", default = "accuracy",
		help = "the tag of row to be extracted (default: accuracy)")
	ap.add_argument("--without-png", action = "store_true",
		help = "do not plot (default: off)")
	args = ap.parse_args()
	return args


def get_results_by_rowtag(fname, rowtag, delimiter = "\t"):
	if not os.path.isfile(fname):
		warnings.warn("file '%s' does not exists, skipping" % fname)
		return None
	with open(fname, "r") as fh:
		lines = fh.read().splitlines()
	for l in lines:
		l = l.split(delimiter)
		if l[0] == rowtag:
			return numpy.asarray(l[1:], dtype = float)
	warnings.warn("no rowtag '%s' found in file '%s'" % (rowtag, fname))
	return None


def load_and_combine_results(prefix, rowtag, delimiter, reduc_dims_to):
	# load/extract results <rowtag> from multiple files
	# organize 1st dim as classification method
	# 2nd dim as dim reduction methods
	# 3nd dim as folds from cross validation
	ret = []
	for classifier in SUMMARIZE_CLASSIFIERS:
		r = []
		for dimreduc in SUMMARIZE_DIMREDUC:
			try:
				ifile = "%s.%s.dr_%s.10_fold.overall.txt"\
					% (prefix, classifier["id"], dimreduc["id"] % reduc_dims_to)
			except TypeError as e:
				ifile = "%s.%s.dr_%s.10_fold.overall.txt"\
					% (prefix, classifier["id"], dimreduc["id"])
			r.append(get_results_by_rowtag(ifile, rowtag, delimiter))
		ret.append(r)
	return ret


def compute_mean_stdev(data):
	"""
	compute mean and stdev of a 3-d array along its third dimension
	some cell may appear a None
	"""
	mean  = [[(numpy.nan if c is None else c.mean()) for c in r] for r in data]
	stdev = [[(numpy.nan if c is None else c.std())  for c in r] for r in data]
	mean = numpy.asarray(mean, dtype = float)
	stdev = numpy.asarray(stdev, dtype = float)
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


def save_tableplot(fname, mean, stdev, reduc_dims_to, title = ""):
	# lazy load
	from matplotlib import pyplot
	# layout
	figure = pyplot.figure(figsize = (10, 4))
	axes = figure.add_axes([0.15, 0.10, 0.80, 0.70]) # tabular
	cmax = figure.add_axes([0.85, 0.10, 0.06, 0.70]) # colorbar
	# hide axes box
	for ax in [axes, cmax]:
		for sp in ax.spines.values():
			sp.set_visible(False)
	# hide ticks
	axes.tick_params(
		top = False, labeltop = False,
		bottom = False, labelbottom = False,
		left = False, labelleft = False,
		right = False, labelright = False)
	cmax.tick_params(
		top = False, labeltop = False,
		bottom = False, labelbottom = False,
		left = False, labelleft = False,
		right = True, labelright = True)
	# invert y axis
	axes.invert_yaxis()
	# plot mean
	alpha = 0.8
	cmap = pyplot.get_cmap("bwr")
	axes.pcolor(mean, alpha = alpha, cmap = cmap, vmin = -1.0, vmax = 1.0)
	# cell text labels 
	nr, nc = mean.shape
	for i in range(nr):
		for j in range(nc):
			value = mean[i, j]
			text = "N/A" if numpy.isnan(value) else ("%.3f" % value)
			# if blue is dark, use white; else black
			color = "#FFFFFF" if value > 0.6 else "#000000"
			axes.text(x = j + 0.5, y = i + 0.5, s = text,
				color = color, fontsize = 12, fontweight = "bold",
				horizontalalignment = "center", verticalalignment = "center")
	# x labels
	for i, dimreduc in enumerate(SUMMARIZE_DIMREDUC):
		try:
			col_header = dimreduc["display_name"] % reduc_dims_to
		except TypeError as e:
			col_header = dimreduc["display_name"]
		axes.text(x = i + 0.5, y = -0.3, s = col_header,
			clip_on = False, color = "k", fontsize = 14,
			horizontalalignment = "center", verticalalignment = "center")
	# y labels
	for i, classifier in enumerate(SUMMARIZE_CLASSIFIERS):
		axes.text(x = -0.1, y = i + 0.5, s = classifier["display_name"],
			clip_on = False, color = "k", fontsize = 14,
			horizontalalignment = "right", verticalalignment = "center")
	# axis limits
	axes.set_xlim(0, mean.shape[1])
	axes.set_xlim(0, mean.shape[0])
	# color bar
	gradient = numpy.linspace(0, 1, 256).reshape(-1, 1)
	cmax.imshow(gradient, alpha = alpha, aspect = "auto", cmap = cmap)
	cmax.invert_yaxis()
	# yticks
	cmax_ymin, cmax_ymax = 128, 256
	cmax_yrange = cmax_ymax - cmax_ymin
	yticks = numpy.linspace(cmax_ymin, cmax_ymax, 5)
	yticklabels = ["%.1f" % ((i - cmax_ymin) / cmax_yrange) for i in yticks]
	cmax.set_ylim(cmax_ymin, cmax_ymax)
	cmax.set_yticks(yticks)
	cmax.set_yticklabels(yticklabels)
	# misc
	figure.suptitle(title, fontsize = 18, verticalalignment = "top")
	pyplot.savefig(fname)
	#pyplot.show()
	pyplot.close()
	return


def main():
	args = get_args()
	data = load_and_combine_results(args.input_prefix, args.row_tag, args.delimiter,\
		args.reduc_dims_to)
	# calculate mean and std
	mean, std = compute_mean_stdev(data)
	# output
	os.makedirs(os.path.dirname(args.output_prefix), exist_ok = True)
	savetxt(args.output_prefix + ".tsv", mean, std, args.delimiter)
	if not args.without_png:
		title = args.input_prefix + ", " + args.row_tag + " summary"
		save_tableplot(args.output_prefix + ".png", mean, std, args.reduc_dims_to, title = title)
	return


if __name__ == "__main__":
	main()