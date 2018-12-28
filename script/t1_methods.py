#!/usr/bin/env python3

import numpy
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.svm
import argparse
from matplotlib import pyplot
import os


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data", type = str, metavar = "tsv", required = True,
		help = "sample data file tsv (required)")
	ap.add_argument("--meta", type = str, metavar = "tsv", required = True,
		help = "metadata file tsv (required)")
	ap.add_argument("--model", type = str, required = True,
		choices = ["gnb", "lr", "lda", "svm_lin", "svm_rbf"],
		help = "choice of fitting model (required)")
	ap.add_argument("--pca", type = str, metavar = "int|none", default = "none",
		help = "apply dimension reduction with PCA (none or any positive integer, default: none)")
	ap.add_argument("--cv-folds", type = int, metavar = "int", default = 10,
		help = "n-fold cross validation (default: 10)")
	args = ap.parse_args()
	# check args
	if args.cv_folds <= 0:
		raise ValueError("--cv-folds must be positive integer")
	if args.pca != "none":
		try:
			args.pca = int(args.pca)
			if args.pca <= 0:
				raise Exception()
		except:
			raise ValueError("--pca must be none or positive integer")
	return args


def load_data(dataf, metaf):
	data = numpy.loadtxt(dataf, dtype = float, delimiter = "\t")
	with open(metaf, "r") as fh:
		meta = fh.read().splitlines()
	labels = [i.split("\t")[0] for i in meta]
	labels = numpy.asarray(labels, dtype = object)
	return data, labels


def encode_labels(labels):
	le = sklearn.preprocessing.LabelEncoder()
	le.fit(labels)
	return le, le.transform(labels)


def split_bool_mask(bool_vec, n_splits):
	nonzero = numpy.nonzero(bool_vec)[0]
	slice_size = len(nonzero) / n_splits
	ret = []
	for i in range(n_splits):
		mask = numpy.zeros(len(bool_vec), dtype = bool)
		s_start = int(slice_size * i)
		s_end = int(slice_size * (i + 1))
		selects = nonzero[s_start:s_end]
		mask[selects] = True
		ret.append(mask)
	return ret


def cv_equal_classes(labels, n_fold = 10):
	splits = []
	# uniq labels
	uniq_labels = numpy.unique(labels)
	uniq_labels.sort()
	# find slice size (decimal) for each label
	label_splits = [split_bool_mask(labels == l, n_fold) for l in uniq_labels]
	test_masks = [numpy.zeros(len(labels), dtype = bool) for i in range(n_fold)]
	for i, ls in enumerate(label_splits):
		for s, mk in enumerate(ls):
			test_masks[s] = numpy.logical_or(test_masks[s], mk)
	data_masks = [numpy.logical_not(i) for i in test_masks]
	return data_masks, test_masks


def pca_reduce_dimension(data, n_components):
	pca = sklearn.decomposition.PCA(n_components = n_components)
	transformed = pca.fit_transform(data)
	return pca, transformed


def cv_model_fit(model, train_data, train_label):
	if model == "gnb":
		m = sklearn.naive_bayes.GaussianNB()
	elif model == "lr":
		m = sklearn.linear_model.LogisticRegression()
	elif model == "lda":
		m = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
	elif model == "svm_lin":
		m = sklearn.svm.LinearSVC(multi_class = "ovr")
	elif model == "svm_rbf":
		sigma = numpy.median(sklearn.metrics.pairwise.euclidean_distances(train_data))
		gamma = 1.0 / (2 * sigma ** 2)
		m = sklearn.svm.SVC(kernel = "rbf", gamma = gamma)
	else:
		raise RuntimeError("unrecognized model")

	m.fit(train_data, train_label)
	return m

def cv_model_test(fitted_model, test_data, test_label):
	pred_label = fitted_model.predict(test_data)
	ns = {}
	ns["accuracy"] = sklearn.metrics.accuracy_score(test_label, pred_label)
	ns["precision"] = sklearn.metrics.precision_score(test_label, pred_label, average = "micro")
	ns["precision_class"] = sklearn.metrics.precision_score(test_label, pred_label, average = None)
	ns["f"] = sklearn.metrics.f1_score(test_label, pred_label, average = "micro")
	ns["f_class"] = sklearn.metrics.f1_score(test_label, pred_label, average = None)
	#ns["auc"] = sklearn.metrics.roc_auc_score(test_label, pred_label, average = "micro")
	#ns["auc_class"] = sklearn.metrics.roc_auc_score(test_label, pred_label, average = None)
	return ns


def main():
	args = get_args()
	# load data
	data, labels = load_data(args.data, args.meta)

	# preprocessing
	uniq_labels = numpy.unique(labels)
	uniq_labels.sort()
	le, encoded_l = encode_labels(labels)

	# cross validation
	cv_data_masks, cv_test_masks = cv_equal_classes(encoded_l, args.cv_folds)
	if args.pca != "none":
		pca, data = pca_reduce_dimension(data, args.pca)

	# cross validation
	cv_res = []
	for dmsk, tmsk in zip(cv_data_masks, cv_test_masks):
		model = cv_model_fit(args.model, data[dmsk], encoded_l[dmsk])
		_test_res = cv_model_test(model, data[tmsk], encoded_l[tmsk])
		cv_res.append(_test_res)

	# plot
	cmap = pyplot.get_cmap("tab10")
	fig = pyplot.figure(figsize = (12, 4))
	suptitle = "%s, model=%s, PCA=%s, n_fold=%d" % (os.path.basename(args.data),
		args.model, str(args.pca), args.cv_folds)
	fig.suptitle(suptitle)
	ax_sep = fig.add_axes([0.07, 0.25, 0.70, 0.62])
	ax_all = fig.add_axes([0.78, 0.25, 0.05, 0.62])
	# plot separate
	for i, key in enumerate(["precision_class", "f_class"]):
		color = cmap(i)
		for label in uniq_labels:
			el, = le.transform([label])
			data = [r[key][el] for r in cv_res]
			x = el + 0.80 + 0.20 * i
			bplot = ax_sep.boxplot(data, positions = [x], widths = 0.20,
				patch_artist = True,
				showfliers = False, showmeans = True)
			for box in bplot["boxes"]:
				box.set_facecolor(color)

	ax_sep.set_xlim(0, len(uniq_labels) + 1)
	ax_sep.set_xticks(numpy.arange(1, len(uniq_labels) + 1))
	ax_sep.set_xticklabels(uniq_labels, rotation = 270)
	ax_sep.set_ylim(-0.05, 1.05)
	# plot all
	ax_all.tick_params(labelleft = False, left = False,
		right = True, labelright = True)
	figs = []
	for i, key in enumerate(["precision", "f", "accuracy"]):
		color = cmap(i)
		data = [r[key] for r in cv_res]
		x = 0.80 + 0.20 * i
		bplot = ax_all.boxplot(data, positions = [x], widths = 0.20,
			patch_artist = True,
			showfliers = False, showmeans = True)
		for box in bplot["boxes"]:
			box.set_facecolor(color)
		figs.append(bplot["boxes"][0])
	ax_all.set_xlim(0.5, 1.5)
	ax_all.set_xticks([1])
	ax_all.set_xticklabels(["Overall"], rotation = 270)
	ax_all.set_ylim(-0.05, 1.05)
	ax_all.legend(handles = figs, labels = ["Precision", "F-score", "Accuracy"],
		loc = 2, bbox_to_anchor = [1.7, 1.033])
	output = "./image/%s.%s.pca_%s.%d_fold.png" % (os.path.basename(args.data),
		args.model, str(args.pca), args.cv_folds)
	#pyplot.show()
	pyplot.savefig(output)
	pyplot.close()


if __name__ == "__main__":
	main()
