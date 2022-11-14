#!/usr/bin/env python3

import numpy
import torch
# 3rd party
import wuml
# custom lib
from . import base
import pylib.util


@base.ClassifierCollection.register("nn", "neural_network")
@base.ClassifierAbstract.serialize_init(as_name = "nn")
class BasicNeuralNetworkWuML(base.ClassifierAbstract):
	def fit(self, X, Y, *ka, **kw):
		data = wuml.wData(X_npArray=X, Y_npArray=Y, label_type='discrete')
		self.nn = wuml.basicNetwork('CE', data,
			networkStructure = [
				(2000,'relu'),
				(2000,'relu'),
				(2000,'relu'),
				(144,'none')
			],
			Y_dataType = torch.LongTensor,
			max_epoch = 5000,
			learning_rate = 0.001,
		)
		self.nn.train(print_status = False)
		return

	def predict(self, X, Y = None, *ka, **kw):
		if Y is None:
			Y = numpy.zeros(len(X), dtype = int)
		data = wuml.wData(X_npArray = X, Y_npArray = Y, label_type='discrete')
		ret = self.nn.__call__(data,
			output_type = "ndarray",
			out_structural = "1d_labels"
		)
		return ret

