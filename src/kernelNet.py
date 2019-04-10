#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket


sys.path.append('./src')
sys.path.append('./src/dataset_class')
sys.path.append('./src/algorithms')
sys.path.append('./src/helper')
sys.path.append('./src/optimization')
sys.path.append('./tests')

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from data_input import *
from hsic_algorithms import *
from knet import *


class kernelNet(test_base):
	def __init__(self, new_db):
		db = {}

		db['data_name'] = 'kn'
		db['dataset_class'] = data_input
		db['TF_obj'] = knet

		db['compute_error'] = None
		db['store_results'] = None
		db['run_only_validation'] = True
		db['use_delta_kernel_for_U'] = False

		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							
		db['λ'] = 0.1

		# network info
		db["kernel_net_depth"]=3
		db['width_scale'] = 2
		db['kernel_model'] = identity_net
		db['cuda'] = False

		for i in new_db: db[i] = new_db[i]

		test_base.__init__(self, db)

	def parameter_ranges(self):
		W_optimize_technique = [ism]
		repeat_run = list(range(1,20))
		return [W_optimize_technique, repeat_run]

	def train(self):
		db = self.db
		self.HA = hsic_algorithms(db)
		self.HA.run()

		return [ db['Ψx'].data.numpy(), db['U'], db['U_normalized']]

	def eval(self, X):
		db = self.db
		X_var = torch.from_numpy(X)
		X_var = Variable(X_var.type(db['dataType']), requires_grad=False)

		[AE_out, Ψx] = db['knet'](X_var)
		return Ψx.data.numpy()


