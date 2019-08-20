#!/usr/bin/env python3


import argparse
import json
import numpy
#import pylib
import sys


_METRIC_KEYS = {
	"accuracy": "average_accuracy",
	"fscore": "class_fscore",
	"precision": "class_precision",
	"recall": "class_recall",
}


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str,
		help = "input result record file of t2_2level.py json output")
	ap.add_argument("-m", "--metric", type = str, required = True,
		choices = sorted(_METRIC_KEYS.keys()),
		help = "report metric (required)")
	args = ap.parse_args()
	return args


def load_results_from_json(fname):
	with open(fname, "r") as fh:
		return json.load(fh)


def xmean(list_or_float):
	"""
	if argument is list, report mean; if float, direct return;
	"""
	if isinstance(list_or_float, float):
		return list_or_float
	elif isinstance(list_or_float, list):
		return numpy.mean(list_or_float)
	else:
		raise TypeError("xmean(list_or_float): argument must be either list or float")
	return


def _mean_std(values) -> "%.3f (%.2f)":
	assert isinstance(values, list)
	_mean = numpy.mean(values)
	_std = numpy.std(values)
	return "%.3f (%.2f)" % (_mean, _std)


def level1_mean(model_cvs, metric) -> "training, testing":
	assert isinstance(model_cvs, list)
	KEY = _METRIC_KEYS[metric]
	ret = []
	for s in ["training", "testing"]:
		_extracted = [xmean(i[s]["level_1"][KEY]) for i in model_cvs]
		ret.append(_mean_std(_extracted))
	return tuple(ret)


def level2_overall_mean(model_cvs, metric) -> "training, testing":
	assert isinstance(model_cvs, list)
	KEY = _METRIC_KEYS[metric]
	ret = []
	for s in ["training", "testing"]:
		_extracted = [xmean(i[s]["level_2"]["overall"][KEY])\
			for i in model_cvs]
		ret.append(_mean_std(_extracted))
	return tuple(ret)


def level2_per_level1_mean(model_cvs, lv1_label, metric) -> "training, testing":
	assert isinstance(model_cvs, list)
	KEY = _METRIC_KEYS[metric]
	ret = []
	for s in ["training", "testing"]:
		_extracted = [xmean(i[s]["level_2"]["per_lv1"][lv1_label][KEY])\
			for i in model_cvs]
		ret.append(_mean_std(_extracted))
	return tuple(ret)


def summarize_and_print(results, metric, file = sys.stdout):
	# summarize by model: lv1 train-test, lv2-all train-test, lv2-p0 train-test, etc.
	print("\t".join(["model", "lv1_train", "lv1_test",\
		"lv2_overall_train", "lv2_overall_test",\
		"lv2_p0_train", "lv2_p0_test",\
		"lv2_p1_train", "lv2_p1_test",\
		"lv2_p2_train", "lv2_p2_test"]), file = file)
	res = results["evaluation"]
	lv1_train, lv1_test = level1_mean(res, metric)
	lv2_overall_train, lv2_overall_test = level2_overall_mean(res, metric)
	lv2_p0_train, lv2_p0_test = level2_per_level1_mean(res, 0, metric)
	lv2_p1_train, lv2_p1_test = level2_per_level1_mean(res, 1, metric)
	lv2_p2_train, lv2_p2_test = level2_per_level1_mean(res, 2, metric)
	print("\t".join([results["model"], lv1_train, lv1_test,
		lv2_overall_train, lv2_overall_test,
		lv2_p0_train, lv2_p0_test,
		lv2_p1_train, lv2_p1_test,
		lv2_p2_train, lv2_p2_test]), file = file)
	return


def main():
	args = get_args()
	results = load_results_from_json(args.input)
	summarize_and_print(results, args.metric)
	return


if __name__ == "__main__":
	main()
