#!/usr/bin/env python3

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from basic_dataset import *
from linear_supv_dim_reduction import *

class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'rcv'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['dataset_class'] = basic_dataset
		db['TF_obj'] = linear_supv_dim_reduction
		db['compute_error'] = None
		db['store_results'] = None
		db['separate_data_for_validation'] = False

		db['q'] = 80
		db['num_of_clusters'] = 4
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							# rank constraint ratio

		test_base.__init__(self, db)

		#	overwrite default values
		db['center_and_scale'] = True
		db['kernel_type'] = 'polynomial'		# rbf, linear, polynomial

	def basic_run(self):
		prog.remove_tmp_files()
		fname = prog.output_db_to_text()

		call(["./src/hsic_algorithms.py", fname])

prog = test_obj()
prog.basic_run()

