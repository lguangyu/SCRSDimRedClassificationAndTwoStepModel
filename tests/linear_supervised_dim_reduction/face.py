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
		db['data_name'] = 'face'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['dataset_class'] = basic_dataset
		db['TF_obj'] = linear_supv_dim_reduction
		db['W_optimize_technique'] = ism  			# orthogonal_optimization, ism, DimGrowth, grassman
		db['compute_error'] = None
		db['store_results'] = None
		db['separate_data_for_validation'] = False

		db['q'] = 20
		db['num_of_clusters'] = 20
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							# rank constraint ratio

		test_base.__init__(self, db)


	def parameter_ranges(self):
		W_optimize_technique = [ism, DimGrowth, orthogonal_optimization]	
		return [W_optimize_technique]

	def run_10_fold_single(self, indx):
		db = self.db

		db['W_optimize_technique'] = DimGrowth
		db['separate_data_for_validation'] = True
		self.kick_off_single_from_10_fold(indx)


	def run_10_fold(self):
		db = self.db
		W_optimize_technique = [ism, orthogonal_optimization, DimGrowth, grassman]		
		db['separate_data_for_validation'] = True
		#self.gen_10_fold_data()

		for technique in W_optimize_technique:
			db['W_optimize_technique'] = technique
			self.kick_off_each()

prog = test_obj()
prog.collect_10_fold_info(ism)

#prog.run_10_fold()
#prog.run_10_fold_single(1)
#prog.basic_run()
#prog.batch_run()

