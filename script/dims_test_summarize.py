#!/usr/bin/env python3

import argparse
import functools
import glob
import itertools
import json
import matplotlib
import matplotlib.figure
import matplotlib.gridspec
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
	_nd = set() # n_components
	# res_dict is in root[dataset][dimred][classif][train/test][n_cmpnt] order
	res_dict = OrganizedResultDict()
	for i in results:
		res = pylib.result_parsing.SingleLevelJSONCondenser.parse(i, metric)
		_ds.add(res.dataset)
		_dr.add(res.dimreducer)
		_cf.add(res.classifier)
		_nd.add(res.n_components)
		res_dict.add_result(res)
	_nd.remove(None)
	# 'none' should always the first of drs
	if "none" in _dr:
		_dr.remove("none")
		sorted_dr = ["none"] + sorted(_dr)
	return dict(datasets = sorted(_ds), dimreducers = sorted_dr,
		classifiers = sorted(_cf), min_ndim = min(_nd), max_ndim = max(_nd),
		results = res_dict)


################################################################################
# plot logics
class DimsTestPilePlotter(object):
	############################################################################
	# variables
	palette = {
		"traininig":	"#4E7AD9",
		"testing":		"#9971D1",
		"background":	"#E8E8F0",
		"grid":			"#C0C0C0",
	}

	def _mark_refit_flag(func):
		"""
		decorator on methods that changes the flag of refit layout;
		internal use only;
		"""
		def wrapper(self, *ka, **kw):
			# NOTE: set the refit flag MUST before the func call
			self._flag_refit = True
			return func(self, *ka, **kw)
		return wrapper

	def _remove_refit_flag_after(func):
		"""
		decorator on methods that the flag of refit; if flagged, call func then
		remove the flag after; otherwise do nothing;
		"""
		def wrapper(self, *ka, **kw):
			ret = None
			if self._flag_refit:
				ret = func(self, *ka, **kw)
				self._flag_refit = False
			return ret
		return wrapper

	@property
	def datasets(self):
		return self._ds
	@datasets.setter
	@_mark_refit_flag
	def datasets(self, value):
		if not ((value is None) or isinstance(value, list)):
			raise ValueError("datasets must be list, not '%s'"\
				% type(value).__name__)
		self._ds = value
		return

	@property
	def dimreducers(self):
		return self._dr
	@dimreducers.setter
	@_mark_refit_flag
	def dimreducers(self, value):
		if not ((value is None) or isinstance(value, list)):
			raise ValueError("dimreducers must be list, not '%s'"\
				% type(value).__name__)
		self._dr = value
		return

	@property
	def classifiers(self):
		return self._cf
	@classifiers.setter
	@_mark_refit_flag
	def classifiers(self, value):
		if not ((value is None) or isinstance(value, list)):
			raise ValueError("classifiers must be list, not '%s'"\
				% type(value).__name__)
		self._cf = value
		return

	@property
	def results(self):
		return self._rs
	@results.setter
	@_mark_refit_flag
	def results(self, value):
		if not ((value is None) or isinstance(value, OrganizedResultDict)):
			raise ValueError("results must be OrganizedResultDict, not '%s'"\
				% type(value).__name__)
		self._rs = value
		return

	@property
	def min_ndim(self):
		return self._nd_min
	@min_ndim.setter
	def min_ndim(self, value):
		if not isinstance(value, int):
			raise ValueError("min_ndim must be int, not '%s'"\
				% type(value).__name__)
		self._nd_min = value
		return

	@property
	def max_ndim(self):
		return self._nd_max
	@max_ndim.setter
	def max_ndim(self, value):
		if not isinstance(value, int):
			raise ValueError("max_ndim must be int, not '%s'"\
				% type(value).__name__)
		self._nd_max = value
		return

	@_mark_refit_flag
	def __init__(self, **kw):
		# initialize attributes to safe use the properties
		self.datasets		= None
		self.dimreducers	= None
		self.classifiers	= None
		for k, v in kw.items():
			# only set the properties, not otherthings
			if hasattr(type(self), k) and\
				isinstance(getattr(type(self), k), property):
				setattr(self, k, v)
			else:
				raise AttributeError("%s has no property attribute '%s'"\
					% (type(self).__name__, k))
		return

	############################################################################
	# other useful properties frequently used in plotting
	@property
	def nblk(self):
		"""
		number of blocks, is number of datasets
		"""
		return len(self.datasets)
	@property
	def nrow(self):
		"""
		number of rows in a block (dataset plot), is number of classifiers
		"""
		return len(self.classifiers)
	@property
	def ncol(self):
		"""
		number of columns in a block (dataset plot), is number of dimreducers
		"""
		return len(self.dimreducers)

	@property
	def blocks(self):
		"""
		a list of blocks (axes grids) each used to plot a dataset;
		"""
		return self._blks
	def _init_blocks(self):
		self._blks = list()
		return

	class AxesBlock(matplotlib.gridspec.GridSpec):
		"""
		block of axes; each block plots results from a single dataset;
		"""
		@functools.wraps(matplotlib.gridspec.GridSpec.__init__)
		def __init__(self, nrow, ncol, *ka, **kw):
			super(DimsTestPilePlotter.AxesBlock, self)\
				.__init__(nrow, ncol, *ka, **kw)
			# create axes
			self.axes = numpy.empty((nrow, ncol), dtype = object)
			for r in range(nrow):
				for c in range(ncol):
					ax = self.figure.add_subplot(self[r, c])
					self.axes[r, c] = ax
			for ax in self.axes.ravel():
				# call foreign method, maybe not a good idea (hard to override)
				DimsTestPilePlotter.apply_axes_style(ax)
			return

		def get_axes(self, *ka):
			return self.axes.__getitem__(ka)

		def add_row_labels(self, labels):
			for lb, ax in zip(labels, self.axes[:, 0]):
				label_text = ax.text(x = -0.3, y = 0.5, s = lb.upper(),
					fontsize = 14, fontweight = "bold", color = "#000000",
					clip_on = False, rotation = 90, transform = ax.transAxes,
					horizontalalignment = "center",
					verticalalignment = "center")
			return

		def add_column_labels(self, labels):
			for lb, ax in zip(labels, self.axes[0, :]):
				label_text = ax.text(x = 0.5, y = 1.05, s = lb.upper(),
					fontsize = 14, fontweight = "bold", color = "#000000",
					clip_on = False, transform = ax.transAxes,
					horizontalalignment = "center",
					verticalalignment = "bottom")
			return

		def set_yticks(self, ticks, **kw):
			for ax in self.axes.ravel():
				ax.set_yticks(ticks)
			for ax in self.axes[:, 0]:
				ax.tick_params(left = True, labelleft = True, **kw)
			return

		def set_xticks(self, ticks, **kw):
			for ax in self.axes.ravel():
				ax.set_xticks(ticks)
			for ax in self.axes[-1, :]:
				ax.tick_params(bottom = True, labelbottom = True, **kw)
			return

		def set_title(self, title, **kw):
			fig = self.figure
			x = numpy.mean([self.left, self.right])
			y = numpy.mean([1.0, self.top])
			for k, v in dict(transform = fig.transFigure, clip_on = False,
					fontsize = 20, fontweight = "bold", color = "#000000",
					horizontalalignment = "center", verticalalignment = "bottom"
				).items():
				if k not in kw:
					kw[k] = v
			label = fig.text(x, y, title, **kw)
			return

	@classmethod
	def apply_axes_style(cls, axes):
		for sp in axes.spines.values():
			sp.set_visible(False)
		axes.set_facecolor(cls.palette["background"])
		axes.tick_params(left = False, right = False, bottom = False,
			top = False, labelleft = False, labelright = False,
			labelbottom = False, labeltop = False)
		axes.set_ylim(0.0, 1.05)
		axes.grid(linestyle = "-", linewidth = 1.0, color = cls.palette["grid"])
		return

	############################################################################
	# plot functions
	@property
	def figure(self):
		if not hasattr(self, "_fig"):
			setattr(self, "_fig", matplotlib.pyplot.figure())
		return getattr(self, "_fig")

	@_remove_refit_flag_after
	def _setup_layout(self):
		# check required attributes
		if (self.datasets is None) or (self.dimreducers is None)\
			or (self.classifiers is None):
			raise ValueError("must set all below parameters: "\
				+ str(["datasets", "dimreducers", "classifiers"]))
		# clean up old figure
		self.figure.clear()
		self._init_blocks()
		# local
		nblk = self.nblk
		nrow = self.nrow
		ncol = self.ncol
		# margins
		left_margin_inch	= 0.8
		right_margin_inch	= 0.2
		top_margin_inch		= 1.2
		bottom_margin_inch	= 1.0
		# axes dimensions
		axes_width_inch		= 2.0
		axes_height_inch	= 1.5
		axes_inblock_wgap	= 0.1
		axes_inblock_hgap	= 0.1
		# block dimensions
		block_width_inch	= axes_width_inch * ncol\
			+ axes_inblock_wgap * (ncol - 1)
		block_height_inch	= axes_height_inch * nrow\
			+ axes_inblock_hgap * (nrow - 1)
		interblock_wgap		= 0.5
		# figure size and apply to figure
		figure_width_inch	= left_margin_inch + block_width_inch * nblk\
			+ interblock_wgap * (nblk - 1) + right_margin_inch
		figure_height_inch	= bottom_margin_inch + block_height_inch\
			+ top_margin_inch
		self.figure.set_size_inches(figure_width_inch, figure_height_inch)
		# create axes blocks
		for i in range(nblk):
			block_left = (left_margin_inch\
				+ (block_width_inch + interblock_wgap) * i) / figure_width_inch
			block_right = block_left + block_width_inch / figure_width_inch
			block_bottom = bottom_margin_inch / figure_height_inch
			block_top = (bottom_margin_inch + block_height_inch)\
				/ figure_height_inch
			wgap = axes_inblock_wgap / axes_width_inch
			hgap = axes_inblock_hgap / axes_height_inch
			# create and add new block
			new_block = self.AxesBlock(nrow, ncol, figure = self.figure,
				left = block_left, bottom = block_bottom, right = block_right,
				top = block_top, wspace = wgap, hspace = hgap)
			self.blocks.append(new_block)
		return

	def _plot_dim_profile(self, axes, prof):
		assert isinstance(prof, dict), type(prof)
		assert isinstance(axes, matplotlib.axes.Axes), type(axes)
		x = numpy.arange(self.max_ndim + 1)
		for mode in ["traininig", "testing"]:
			y = numpy.full(self.max_ndim + 1, numpy.nan, dtype = float)
			for d, acc in prof[mode].items():
				if d is None:
					# for dr='none' where dimensions is None
					y[self.min_ndim:] = acc
				else:
					y[d] = acc
			color = self.palette[mode]
			axes.plot(x, y, linestyle = "-", linewidth = 2.0, color = color)
		return

	def plot(self, fname):
		self._setup_layout()
		# plot each axes
		for i_ds, i_dr, i_cf in itertools.product(
				*map(lambda x: range(x), [self.nblk, self.ncol, self.nrow])):
			# get data and axes to plot onto
			ds = self.datasets[i_ds]
			dr = self.dimreducers[i_dr]
			cf = self.classifiers[i_cf]
			profile = self.results[ds][dr][cf]
			axes = self.blocks[i_ds].get_axes(i_cf, i_dr)
			# plot
			self._plot_dim_profile(axes, profile)
			axes.set_xlim(0, self.max_ndim + 2)
		# misc
		self.blocks[0].add_row_labels(self.classifiers)
		for blk, ds in zip(self.blocks, self.datasets):
			blk.add_column_labels(self.dimreducers)
			blk.set_xticks([2, 5, 10, 20, 30, 40])
			blk.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
			blk.set_title(ds.upper())

		# legend
		leg_handles = list()
		for mode in ["traininig", "testing"]:
			leg_handles.append(matplotlib.lines.Line2D([], [], linestyle = "-",
				linewidth = 2.0, color = self.palette[mode], label = mode))
		self.figure.legend(handles = leg_handles, loc = "lower center",
			ncol = 2, fontsize = 18, frameon = False)

		# save
		matplotlib.pyplot.savefig(fname, dpi = 100)
		matplotlib.pyplot.close()
		return


def main():
	args = get_args()
	results = organize_results(load_all_results(args.input), args.metric)
	# output
	with pylib.util.file_io.get_fh(args.output, "w") as fp:
		json.dump(results, fp)
	# plot
	if args.plot:
		plotter = DimsTestPilePlotter(**results)
		plotter.plot(args.plot)
	return


if __name__ == "__main__":
	main()
