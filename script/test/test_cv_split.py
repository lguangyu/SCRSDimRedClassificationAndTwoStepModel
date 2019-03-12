#!/usr/bin/env python3

import unittest
import numpy
# tested module
import custom_pylib.cv_split


class TestEvenSplitCrossValidation(unittest.TestCase):
	def test_get_label_split_mask(self):
		test_class = custom_pylib.cv_split.EvenSplitCrossValidation
		# can be even splitted
		labels = [False, False, True, True]
		result = test_class._get_label_split_mask(labels, True, 2)
		expect = [[2], [3]]
		for res, exp in zip(result, expect):
			self.assertTrue(numpy.array_equal(res, exp))
		# must uneven split
		labels = [False, False, True, True, True, True, True]	
		result = test_class._get_label_split_mask(labels, True, 3)
		expect = [[2, 3], [4, 5], [6]]
		for res, exp in zip(result, expect):
			self.assertTrue(numpy.array_equal(res, exp))

	def test_generate_test_masks(self):
		test_class = custom_pylib.cv_split.EvenSplitCrossValidation
		data = numpy.arange(10)
		# label should be ok with missing labels
		# though we explicitly require it
		labels = [1] * 2 + [4] * 3 + [6] * 4 + [10] * 1
		result = test_class._generate_test_masks(data, labels, 3)
		exp_test_masks = [
			[True, False, True, False, False, True, True, False, False, True],
			[False, True, False, True, False, False, False, True, False, False],
			[False, False, False, False, True, False, False, False, True, False],
		]
		for res, exp in zip(result, exp_test_masks):
			self.assertTrue(numpy.array_equal(res, exp))

	def test_get_split(self):
		data = numpy.arange(10)
		labels = [0] * 2 + [1] * 3 + [2] * 4 + [3] * 1
		escv = custom_pylib.cv_split.EvenSplitCrossValidation(data, labels, 4)
		exp_test_masks = [
			[True, False, True, False, False, True, False, False, False, True],
			[False, True, False, True, False, False, True, False, False, False],
			[False, False, False, False, True, False, False, True, False, False],
			[False, False, False, False, False, False, False, False, True, False],
		]
		for i, exp in enumerate(exp_test_masks):
			self.assertTrue(numpy.array_equal(escv.get_split(i, True)[1], exp))
		exp_train_data = [numpy.nonzero(numpy.logical_not(i))[0]\
			for i in exp_test_masks]
		exp_test_data = [numpy.nonzero(i)[0] for i in exp_test_masks]
		exp_train_label = [numpy.compress(numpy.logical_not(i), labels)\
			for i in exp_test_masks]
		exp_test_label = [numpy.compress(i, labels)\
			for i in exp_test_masks]
		for i, folds in enumerate(escv):
			train_data, test_data, train_label, test_label = folds
			#
			self.assertTrue(numpy.array_equal(train_data, exp_train_data[i]))
			self.assertTrue(numpy.array_equal(test_data, exp_test_data[i]))
			self.assertTrue(numpy.array_equal(train_label, exp_train_label[i]))
			self.assertTrue(numpy.array_equal(test_label, exp_test_label[i]))


if __name__ == "__main__":
	unittest.main()
