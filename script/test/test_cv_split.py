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
		expect = [
			[False, False, True, False, False, True, False, False, False, False],
			[True, False, False, True, False, False, True, False, False, False],
			[False, True, False, False, True, False, False, True, True, True],
		]
		for res, exp in zip(result, expect):
			self.assertEqual(res.tolist(), exp)

	def test_get_split(self):
		data = numpy.arange(10)
		labels = [0] * 2 + [1] * 3 + [2] * 4 + [3] * 1
		escv = custom_pylib.cv_split.EvenSplitCrossValidation(data, labels, 4)
		expect = [
			[False, False, False, False, False, True, False, False, False, False],
			[True, False, True, False, False, False, True, False, False, False],
			[False, False, False, True, False, False, False, True, False, False],
			[False, True, False, False, True, False, False, False, True, True],
		]
		for i, exp in enumerate(expect):
			self.assertEqual(escv.get_split(i, True)[1].tolist(), exp)
		expect_val = [
			[5],
			[0, 2, 6],
			[3, 7],
			[1, 4, 8, 9],
		]
		i = 0
		for train_data, test_data, train_label, test_label in escv:
			self.assertEqual(test_data.tolist(), expect_val[i])
			i = i + 1


if __name__ == "__main__":
	unittest.main()
