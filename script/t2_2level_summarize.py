#!/usr/bin/env python3


import abc
import argparse
import json
import numpy
#import pylib
import sys
import traceback
import warnings


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str,
		help = "input result record file of t2_2level.py json output")
	args = ap.parse_args()
	return args


def load_results(fname):
	with open(fname, "r") as fh:
		return json.load(fh)


def fold_average_accuracy(model_results):
	ret = {}
	accuracies = [i["training evaluation"]["accuracy"] for i in model_results]
	ret["training_mean"] = numpy.mean(accuracies)
	ret["training_std"] = numpy.std(accuracies)
	accuracies = [i["testing evaluation"]["accuracy"] for i in model_results]
	ret["testing_mean"] = numpy.mean(accuracies)
	ret["testing_std"] = numpy.std(accuracies)
	return ret


def main():
	args = get_args()
	results = parse_results_txt(args.input)
	res_accuracy = {k: fold_average_accuracy(v)\
		for k, v in results["models"].items()}
	res_accuracy = list(res_accuracy.items())
	res_accuracy.sort(key = lambda x: x[1]["testing_mean"], reverse = True)
	for model, res in res_accuracy:
		#print("%s\t, %.3f (%.2f), %.3f (%.2f), 
		print(model, res)
	return





if __name__ == "__main__":
	main()
