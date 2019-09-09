#!/usr/bin/env python3

import argparse
import itertools
import matplotlib
import matplotlib.cm
import matplotlib.pyplot
import numpy
import sklearn.decomposition
import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "-d", "--dataset", type = str, required = True,
		metavar = "dataset",
		choices = pylib.DatasetCollection.get_registered_keys(),
		help = "the dataset to analyze (required); choices: "\
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


def _setup_layout(n_classes) -> dict:
	ret = dict()
	# margins
	left_margin_inch	= 0.2
	right_margin_inch	= 0.2
	bottom_margin_inch	= 0.2
	top_margin_inch		= 0.2
	# class plot axes grid
	ncols	= 6
	nrows	= (n_classes + ncols - 1) // ncols
	class_axes_width_inch	= 1.2
	class_axes_height_inch	= class_axes_width_inch
	class_axes_wspace_inch	= 0.2
	class_axes_hspace_inch	= 0.2
	class_axes_grid_width_inch	= ncols * class_axes_width_inch\
		+ (ncols - 1) * class_axes_wspace_inch
	class_axes_grid_height_inch	= nrows * class_axes_height_inch\
		+ (nrows - 1) * class_axes_hspace_inch
	# dimensions of eigenvalues axes
	ylabel_pad_inch			= 0.6 # applied on axes with yaxis label
	grid_eig_axes_gap_inch	= 0.5
	eig_axes_width_inch		= class_axes_grid_width_inch - ylabel_pad_inch
	eig_axes_height_inch	= 1.5
	eig_axes_left_inch		= left_margin_inch + ylabel_pad_inch
	eig_axes_bottom_inch	= bottom_margin_inch + class_axes_grid_height_inch\
		+ grid_eig_axes_gap_inch
	# dimensions of pca plot
	eig_pca_axes_gap_inch	= 0.5
	pca_legend_width_inch	= 1.5
	pca_legend_gap_inch		= 0.1
	pca_axes_width_inch		= eig_axes_width_inch - pca_legend_width_inch\
		- pca_legend_gap_inch # align to eigenvalue axes
	pca_axes_height_inch	= pca_axes_width_inch
	pca_axes_left_inch		= eig_axes_left_inch # align to eigenvalue axes
	pca_axes_bottom_inch	= eig_axes_bottom_inch + eig_axes_height_inch\
		+ eig_pca_axes_gap_inch
	# more calculations on pca legend
	# these coords are using to pca_axes.transAxes
	pca_legend_left		= pca_legend_gap_inch / pca_axes_width_inch + 1.0
	pca_legend_width	= pca_legend_width_inch / pca_axes_width_inch
	# dimensions of figure
	figure_width_inch		= left_margin_inch + class_axes_grid_width_inch\
		+ right_margin_inch
	figure_height_inch		= bottom_margin_inch + class_axes_grid_height_inch\
		+ grid_eig_axes_gap_inch + eig_axes_height_inch\
		+ eig_pca_axes_gap_inch + pca_axes_height_inch + top_margin_inch
	
	# create figure
	figure = matplotlib.pyplot.figure(
		figsize = (figure_width_inch, figure_height_inch))
	# create class grid axes
	class_axes_list = numpy.empty((nrows, ncols), dtype = object)
	for r, c in itertools.product(range(nrows), range(ncols)):
		# left and bottom coords in inches
		_left	= left_margin_inch\
			+ c * (class_axes_width_inch + class_axes_wspace_inch)
		_bottom	= bottom_margin_inch\
			+ (nrows - r - 1) * (class_axes_height_inch + class_axes_hspace_inch)
		axes = figure.add_axes([
			_left / figure_width_inch, _bottom / figure_height_inch,
			class_axes_width_inch / figure_width_inch,
			class_axes_height_inch / figure_height_inch,
		])
		axes.set_visible(False)
		class_axes_list[r, c] = axes
	# create eigenvalue axes
	eig_axes = figure.add_axes([
		eig_axes_left_inch / figure_width_inch, # left
		eig_axes_bottom_inch / figure_height_inch, # bottom
		eig_axes_width_inch / figure_width_inch, # width
		eig_axes_height_inch / figure_height_inch, # height
	])
	# create pca axes
	pca_axes = figure.add_axes([
		pca_axes_left_inch / figure_width_inch,
		pca_axes_bottom_inch / figure_height_inch,
		pca_axes_width_inch / figure_width_inch,
		pca_axes_height_inch / figure_height_inch,
	])

	# return dict
	return {
		"figure": figure,
		"pca_axes": pca_axes,
		"pca_legend": {
			"left": pca_legend_left,
			"width": pca_legend_width,
		},
		"eig_axes": eig_axes,
		"class_axes_list": class_axes_list,
	}



def plot_dataset_pca(plot_file, dataset, n_components = 100):
	if not isinstance(dataset, pylib.dataset.SingleLabelDataset):
		raise ValueError("dataset must be SingleLabelDataset, but the specified"
			" '%s' is not" % args.dataset)
	# get unique labels in dataset
	uniq_label = numpy.unique(dataset.text_label)
	uniq_label.sort()

	# run pca
	pca = sklearn.decomposition.PCA(n_components = n_components)
	transformed	= pca.fit_transform(dataset.data)
	pc_x, pc_y	= transformed[:, :2].T # we only plot first 2 pc's
	eigvals		= pca.explained_variance_ratio_[:n_components]

	# setup for plot
	layout = _setup_layout(len(uniq_label))
	# colors for each class
	colors = list(matplotlib.cm.get_cmap("Set1").colors)\
		+ list(matplotlib.cm.get_cmap("Dark2").colors)\
		+ list(matplotlib.cm.get_cmap("Set2").colors)

	# plot pca
	# style axes
	axes = layout["pca_axes"]
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_facecolor("#E8E8F0")
	axes.grid(linestyle = "-", linewidth = 1.0, color = "#FFFFFF")
	# plot
	handles = []
	for label, color in zip(uniq_label, colors):
		mask = (dataset.text_label == label)
		handles.append(axes.scatter(pc_x[mask], pc_y[mask],
			marker = "o", s = 25, facecolor = "#FFFFFF40", edgecolor = color,
			zorder = 2, label = label))
	# pca plot misc
	axes.set_xlabel("PC1: %.1f%%" % (eigvals[0] * 100))
	axes.set_ylabel("PC2: %.1f%%" % (eigvals[1] * 100))
	axes.legend(handles = handles,
		bbox_to_anchor = [layout["pca_legend"]["left"], -0.01,
			layout["pca_legend"]["width"], 1.02] # -0.01 -> 1.01 in transAxis
	)

	# eigenvalue plot
	# style axes
	axes = layout["eig_axes"]
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_facecolor("#E8E8F0")
	axes.grid(linestyle = "-", linewidth = 1.0, color = "#FFFFFF", axis = "y")
	# plot
	x = numpy.arange(len(eigvals)) + 1
	for i, v in enumerate(eigvals):
		axes.plot([i + 1, i + 1], [0, v], clip_on = True, linestyle = "-",
			linewidth = 2.0, color = "#637094")
	# eigenvalue plot misc
	axes.set_xlim(0, len(eigvals) + 1 )
	axes.set_ylim(0, max(eigvals) * 1.1)
	axes.set_ylabel("explained variance ratio")

	# class plot
	for axes, label in zip(layout["class_axes_list"].ravel(), uniq_label):
		# style axes
		axes.set_visible(True)
		for sp in axes.spines.values():
			sp.set_visible(False)
		axes.set_facecolor("#E8E8F0")
		axes.tick_params(bottom = False, top = False, left = False,
			right = False, labelbottom = False, labeltop = False,
			labelleft = False, labelright = False)
		# highlight this class with colorized
		hl_color = "#DB4E27" # highlighed class
		rs_color = "#9DA6C2" # everything else
		mask = (dataset.text_label == label)
		axes.scatter(pc_x[mask], pc_y[mask],
			s = 15, facecolor = "#FFFFFF", edgecolor = hl_color, zorder = 5)
		reverse_mask = numpy.logical_not(mask)
		# plot anything else as gray
		axes.scatter(pc_x[reverse_mask], pc_y[reverse_mask],
			s = 15, facecolor = rs_color, edgecolor = rs_color, zorder = 4)
		# misc
		axes.text(x = 0.95, y = 0.95, s = label,
			fontsize = 10, transform = axes.transAxes, zorder = 6,
			horizontalalignment = "right", verticalalignment = "top")

	# save figure
	matplotlib.pyplot.savefig(plot_file, dpi = 300)
	return


def main():
	args = get_args()
	# create dataset
	dataset = pylib.DatasetCollection.get_dataset(args.dataset)
	# make plot
	plot_dataset_pca(args.plot, dataset)
	return


if __name__ == "__main__":
	main()
	#args = get_args()
	#data, labels = load_data(args.data, args.meta)
	##
	#fig = pyplot.figure(figsize = (10, 18))

	#plot_pca(fig, data, labels)
	##pyplot.show()
	#output = "image/" + os.path.basename(args.data) + ".pca.png"
	#pyplot.savefig(output)
	#pyplot.close()
