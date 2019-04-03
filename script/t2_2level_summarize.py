#!/usr/bin/env python3


import abc
import argparse
import numpy
#import pylib
import sys
import traceback
import warnings


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str,
		help = "input result record file of t2_2level.py txt output")
	args = ap.parse_args()
	return args


class StructuredDataParserBase(abc.ABC):
	@property
	def curr_subparser(self):
		return self._curr_subparser
	@curr_subparser.setter
	def curr_subparser(self, value):
		if isinstance(value, StructuredDataParserBase):
			self._curr_subparser = value
			return
		raise TypeError("curr_subparser must be StructuredDataParserBase")

	def __init__(self, *ka, **kw):
		super(StructuredDataParserBase, self).__init__(*ka, **kw)
		return

	@abc.abstractmethod
	def parse_handler(self, text):
		return NotImplemented

	def parse(self, text):
		"""
		attemp to parse a chunk of text locally, if not handled, pass to the
		current subparser
		"""
		handled = self.parse_handler(text)
		if not handled:
			# handled locally
			return
		elif handled is NotImplemented:
			# pass to curr_subparser
			return self.curr_subparser.parse(text)
		else:
			raise ValueError("parse_handler must return either None or NotImplemented")

	def set_subparser_from_factory(self, factory, *ka, **kw):
		"""
		create a subparser by calling factory(*ka, **kw) and set current subparser
		to the newly created parser
		"""
		subparser = factory(*ka, **kw)
		self.curr_subparser = subparser
		return subparser


class TwoLevelModelResults(dict, StructuredDataParserBase):
	def __init__(self, *ka, **kw):
		super(TwoLevelModelResults, self).__init__(*ka, **kw)
		self["models"] = dict()
		return

	def parse_handler(self, line):
		label_tags = ["lv1-labels", "lv2-labels"]
		for tag in label_tags:
			if tag in line:
				self[tag] = line.split("\t")[1:]
				return
		if "model:\t" in line:
			model = line.split("\t")[1] # "lda+gnb/lsdr+lda" etc.
			sp = self.set_subparser_from_factory(TwoLevelModelResultModel)
			self["models"][model] = sp
			return
		return NotImplemented


class TwoLevelModelResultModel(list, StructuredDataParserBase):
	def parse_handler(self, line):
		if "fold" in line:
			sp = self.set_subparser_from_factory(TwoLevelModelResultModelCVRound)
			self.append(sp)
			return
		return NotImplemented


class TwoLevelModelResultModelCVRound(dict, StructuredDataParserBase):
	def parse_handler(self, line):
		if line in ["training evaluation:", "testing evaluation:"]:
			sp = self.set_subparser_from_factory(\
				TwoLevelModelResultModelCVRoundResSet)
			self[line[:-1]] = sp # chomp the trailing ':'
			return
		return NotImplemented


class TwoLevelModelResultModelCVRoundResSet(dict, StructuredDataParserBase):
	def parse_handler(self, line):
		splitted = line.split("\t")
		#assert splitted[0][-1] == ":", splitted[0]
		# split tag and data
		tag = splitted[0].rstrip(":")
		data = splitted[1:]
		if not data:
			raise ValueError("line '%s' is badly formatted" % line)
		elif len(data) == 1:
			self[tag] = float(data[0]) # a single value
			return
		else:
			self[tag] = [float(i) for i in data] # a list of values
			return
		return NotImplemented


def parse_results_txt(fname):
	parser = TwoLevelModelResults()
	with open(fname, "r") as fh:
		for i, line in enumerate(fh):
			# remove EOL
			line = line.rstrip()
			try:
				handled = parser.parse(line)
			except Exception as e:
				traceback.print_exc(file = sys.stderr)
				print("above error occurred when handling line %d\n\t%s"\
					% (i, line), file = sys.stderr)
				exit(1)
			if handled is NotImplemented:
				raise RuntimeError("cannot handle line %d:\n\t%s" % (i, line))
	return parser


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
