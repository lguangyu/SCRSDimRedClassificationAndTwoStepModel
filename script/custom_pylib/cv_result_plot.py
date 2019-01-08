#!/usr/bin/env python3

import numpy
from matplotlib import pyplot


BOXPLOT_MEAN_STYLE = dict(linewidth = 2.0, color = "#000000")
BOXPLOT_MEDIAN_STYLE = dict(linewidth = 1.5, color = "#FF8080")


def boxplot(self, prefix, uniq_labels, label_encoder, title = ""):
	"""
	method function defined outside CrossValidation class;
	"""
	result = self.get_result()
	# result structure: result[#fold]{key}{...}([#label])
	cmap = pyplot.get_cmap("Set3")

	# plot
	figure, axes = pyplot.subplots(nrows = 1, ncols = 2, sharey = True,
		figsize = (12, 4))
	ax_cls, ax_avg = axes
	figure.suptitle(title)

	# adjust axes position
	ax_cls.set_position([0.07, 0.25, 0.70, 0.62]) # each class (left)
	ax_avg.set_position([0.78, 0.25, 0.05, 0.62]) # overall (right)

	# plot each class (left)
	for i, key in enumerate(self._CLASSES_KEYS_):
		color = cmap(i)
		for label in uniq_labels:
			enc_label, = label_encoder.transform([label])
			# reorganize data into vectors
			data = [r[key]["classes"][enc_label] for r in result]
			# horizontal position of boxes
			x = enc_label + 0.78 + 0.22 * i
			# boxplot
			bplot = ax_cls.boxplot(data, positions = [x], widths = 0.18,
				patch_artist = True, showfliers = False,
				showmeans = True, meanline = True,
				meanprops = BOXPLOT_MEAN_STYLE,
				medianprops = BOXPLOT_MEDIAN_STYLE)
			# color boxes face
			for box in bplot["boxes"]:
				box.set_facecolor(color)
	# finalize
	ax_cls.set_xlim(0, len(uniq_labels) + 1)
	ax_cls.set_xticks(numpy.arange(1, len(uniq_labels) + 1))
	ax_cls.set_xticklabels(uniq_labels, rotation = 270,
		horizontalalignment = "center", verticalalignment = "top")
	ax_cls.set_ylim(-0.02, 1.05)

	# plot overall average (right)
	ax_avg.tick_params(labelleft = False, left = False,
		right = True, labelright = True)
	figs = [] # used for legends
	for i, key in enumerate(self._OVERALL_KEYS_):
		color = cmap(i)
		# reorganize data into vectors
		data = [r[key]["overall"] for r in result]
		# horizontal position of boxes
		x = 0.78 + 0.22 * i
		# boxplot
		bplot = ax_avg.boxplot(data, positions = [x], widths = 0.18,
			patch_artist = True, showfliers = False,
			showmeans = True, meanline = True,
			meanprops = BOXPLOT_MEAN_STYLE,
			medianprops = BOXPLOT_MEDIAN_STYLE)
		for box in bplot["boxes"]:
			box.set_facecolor(color)
		figs.append(bplot["boxes"][0])
	# finalize
	ax_avg.set_xlim(0.6, 1.4)
	ax_avg.set_xticks([1])
	ax_avg.set_xticklabels(["Average"], rotation = 270,
		horizontalalignment = "center", verticalalignment = "top")
	#ax_avg.set_ylim(-0.02, 1.05) # sharey
	ax_avg.legend(handles = figs, labels = ["Precision", "F-score", "Accuracy"],
		loc = 2, bbox_to_anchor = [1.7, 1.033])

	# save png
	png = "%s.png" % prefix
	#pyplot.show()
	pyplot.savefig(png)
	pyplot.close()
