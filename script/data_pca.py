#!/usr/bin/env python3

import numpy
from matplotlib import pyplot
import sklearn.decomposition
import sys
import os
import argparse


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data", type = str, required = True, metavar = "tsv",
		help = "data file")
	ap.add_argument("-m", "--meta", type = str, required = True, metavar = "tsv",
		help = "meta data file")
	return ap.parse_args()


def load_data(dfile, mfile):
	data = numpy.loadtxt(dfile, delimiter = "\t", dtype = float)
	with open(mfile, "r") as fh:
		meta = fh.read().splitlines()
	labels = numpy.asarray([i.split("\t")[0] for i in meta], dtype = object)
	return data, labels


def add_sep_pca_axes(figure, index):
	row = index // 6
	col = index % 6
	left = 0.06 + col * 0.15
	bottom = 0.030 + (4 - row) * 0.08333
	axes = figure.add_axes([left, bottom, 0.13, 0.07222])
	axes.tick_params(bottom = False, top = False, left = False, right = False,
		labelbottom = False, labeltop = False, labelleft = False, labelright = False)
	return axes


def plot_pca(figure, data, labels):
	uniq_labels = numpy.unique(labels)
	uniq_labels.sort()
	# pca
	pca = sklearn.decomposition.PCA(n_components = 100)
	transformed = pca.fit_transform(data)
	tx, ty = transformed[:, :2].T # first two pc's
	eigvals = pca.explained_variance_[:100]

	# plot
	colors = list(pyplot.get_cmap("Set1").colors) + \
		list(pyplot.get_cmap("Dark2").colors) + \
		list(pyplot.get_cmap("Set2").colors)

	# pca plot
	axes_pca = fig.add_axes([0.10, 0.58, 0.70, 0.39])
	figs = []
	for label, color in zip(uniq_labels, colors):
		mask = (labels == label)
		x = tx[mask]
		y = ty[mask]
		f = axes_pca.scatter(x, y, marker = "o", s = 25,
			facecolor = "#FFFFFF40", edgecolor = color,
			label = label)
		figs.append(f)
	axes_pca.axhline(0.0, linewidth = "0.5", linestyle = "-", color = "#808080")
	axes_pca.axvline(0.0, linewidth = "0.5", linestyle = "-", color = "#808080")
	axes_pca.set_xlabel("%d%%" % (pca.explained_variance_ratio_[0] * 100))
	axes_pca.set_ylabel("%d%%" % (pca.explained_variance_ratio_[1] * 100))
	axes_pca.legend(handles = figs, loc = 2,
		bbox_to_anchor = [1.05, 1.01])
	# eig plot
	axes_eig = fig.add_axes([0.10, 0.47, 0.85, 0.08])
	x = numpy.arange(len(eigvals)) + 1
	axes_eig.scatter(x, eigvals, s = 15, marker = "o", clip_on = False,
		facecolor = "#FFFFFF40", edgecolor = "#4040FF")
	axes_eig.set_xlim(0, max(x) + 1)
	axes_eig.set_ylim(0, max(eigvals) * 1.1)
	axes_eig.set_ylabel("Eigenvalue")

	# separate pca plot
	for i, label in enumerate(uniq_labels):
		axes = add_sep_pca_axes(figure, i)
		mask = (labels == label)
		cx, cy = tx[mask], ty[mask]
		nmask = numpy.logical_not(mask)
		nx, ny = tx[nmask], ty[nmask]
		axes.scatter(nx, ny, marker = "o", s = 15,
			facecolor = "#D0D0D0", edgecolor = "#D0D0D0")
		axes.scatter(cx, cy, marker = "o", s = 15,
			facecolor = "#FFFFFF", edgecolor = "#4040FF")
		axes.set_xlabel(label, fontsize = 8)
	return


if __name__ == "__main__":
	args = get_args()
	data, labels = load_data(args.data, args.meta)
	#
	fig = pyplot.figure(figsize = (10, 18))

	plot_pca(fig, data, labels)
	#pyplot.show()
	output = "image/" + os.path.basename(args.data) + ".pca.png"
	pyplot.savefig(output)
	pyplot.close()
