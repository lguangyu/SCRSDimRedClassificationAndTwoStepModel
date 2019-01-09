#!/usr/bin/env python3

import unittest
import numpy
# tested module
import custom_pylib.cv_split


class TestEvenSplitCrossValidation(unittest.TestCase):
	def test_split_bool_vector(self):
		test_class = custom_pylib.cv_split.EvenSplitCrossValidation
		# can be even splitted
		bool_vec = [False, False, True, True]
		result = test_class._split_bool_vector(bool_vec, 2)
		expect = [[2], [3]]
		for res, exp in zip(result, expect):
			self.assertEqual(res, exp)
		# must uneven split
		bool_vec = [False, False, True, True, True, True, True]	
		result = test_class._split_bool_vector(bool_vec, 3)
		expect = [[2], [3, 4], [5, 6]]
		for res, exp in zip(result, expect):
			self.assertEqual(res, exp)

	def test_generate_test_masks(self):
		test_class = custom_pylib.cv_split.EvenSplitCrossValidation
		data = numpy.arange(10)
		# label should be ok with missing labels
		# though we explicitly require it
		labels = [1] * 2 + [4] * 3 + [6] * 4 + [10] * 1
		result = test_class._generate_test_masks(data, labels, 3)
		exp_test_masks = [
			[False, False, True, False, False, True, False, False, False, False],
			[True, False, False, True, False, False, True, False, False, False],
			[False, True, False, False, True, False, False, True, True, True],
		]
		for res, exp in zip(result, exp_test_masks):
			self.assertEqual(res.tolist(), exp)

	def test_get_split(self):
		data = numpy.arange(10)
		labels = [0] * 2 + [1] * 3 + [2] * 4 + [3] * 1
		escv = custom_pylib.cv_split.EvenSplitCrossValidation(data, labels, 4)
		exp_test_masks = [
			[False, False, False, False, False, True, False, False, False, False],
			[True, False, True, False, False, False, True, False, False, False],
			[False, False, False, True, False, False, False, True, False, False],
			[False, True, False, False, True, False, False, False, True, True],
		]
		for i, exp in enumerate(exp_test_masks):
			self.assertEqual(escv.get_split(i, True)[1].tolist(), exp)
		exp_train_data = [
			[0, 1, 2, 3, 4, 6, 7, 8, 9],
			[1, 3, 4, 5, 7, 8, 9],
			[0, 1, 2, 4, 5, 6, 8, 9],
			[0, 2, 3, 5, 6, 7],
		]
		exp_test_data = [
			[5],
			[0, 2, 6],
			[3, 7],
			[1, 4, 8, 9],
		]
		exp_train_label = [
			[0, 0, 1, 1, 1, 2, 2, 2, 3],
			[0, 1, 1, 2, 2, 2, 3],
			[0, 0, 1, 1, 2, 2, 2, 3],
			[0, 1, 1, 2, 2, 2],
		]
		exp_test_label = [
			[2],
			[0, 1, 2],
			[1, 2],
			[0, 1, 2, 3],
		]
		i = 0
		for train_data, test_data, train_label, test_label in escv:
			self.assertEqual(train_data.tolist(), exp_train_data[i])
			self.assertEqual(test_data.tolist(), exp_test_data[i])
			self.assertEqual(train_label.tolist(), exp_train_label[i])
			self.assertEqual(test_label.tolist(), exp_test_label[i])
			i = i + 1


if __name__ == "__main__":
	unittest.main()
