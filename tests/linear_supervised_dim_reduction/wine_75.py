#!/usr/bin/env python3

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from basic_dataset import *
from linear_supv_dim_reduction import *
from ism import *
from orthogonal_optimization import *
from DimGrowth import *
from grassman import *

class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'wine'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['dataset_class'] = basic_dataset
		db['TF_obj'] = linear_supv_dim_reduction
		db['W_optimize_technique'] = ism		# orthogonal_optimization, ism, DimGrowth, grassman

		db['compute_error'] = None
		db['store_results'] = None
		db['separate_data_for_validation'] = False

		db['q'] = 4
		db['num_of_clusters'] = 3
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 0.0							# rank constraint ratio
		db['kernel_type'] = 'rbf'

		test_base.__init__(self, db)


	def parameter_ranges(self):
		W_optimize_technique = [ism, DimGrowth, orthogonal_optimization]	
		#id_10_fold = [0] #range(10)
		#lambda_ratio = [2]
		#random.shuffle(σ_ratio)
	
		return [W_optimize_technique]

	def run_10_fold(self):
		self.db['separate_data_for_validation'] = True
		#self.gen_10_fold_data()
		self.kick_off_each()

prog = test_obj()
#prog.run_10_fold()
prog.basic_run()
#prog.batch_run()

