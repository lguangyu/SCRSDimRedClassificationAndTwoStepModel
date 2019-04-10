#!/usr/bin/env python
# -*- coding: utf-8 -*-

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from basic_dataset import *
from sknet import *
from MLP_autoencoder import *
import torch
from torch.autograd import Variable

class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'spiral'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file
		db['print_debug'] = True
		db['batch_size'] = 100

		db['dataset_class'] = basic_dataset
		db['TF_obj'] = sknet

		db['compute_error'] = None
		db['store_results'] = None
		db['separate_data_for_validation'] = False
		db['use_delta_kernel_for_U'] = False

		db['num_of_clusters'] = 3
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							
		db['λ'] = 0.0

		# network info
		db["kernel_net_depth"]=3
		db['width_scale'] = 2
		db['kernel_model'] = identity_net
		db['cuda'] = False

		test_base.__init__(self, db)


	def parameter_ranges(self):
		W_optimize_technique = [ism]
		repeat_run = list(range(1,20))
		return [W_optimize_technique, repeat_run]


prog = test_obj()
prog.basic_run()
#prog.batch_run()

