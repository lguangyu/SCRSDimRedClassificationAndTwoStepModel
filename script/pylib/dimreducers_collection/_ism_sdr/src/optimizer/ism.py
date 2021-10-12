#!/usr/bin/env python3

import numpy as np
from ..tools import kernel_lib as klib
#from terminal_print import *


class ism():
	def __init__(self, db):
		self.db = db
		self.conv_threshold = 0.01

	def __del__(self):
		pass

	def run(self, old_W, max_rep=200):
		db = self.db

		new_W = old_W
		old_λ = np.random.randn(1,db['q'])

		for i in range(max_rep):
			Φ = db['kernel'].get_Φ(old_W)
			[new_W, new_λ] = klib.eig_solver(Φ, db['q'])

			if self.inner_converge(new_W, old_W, new_λ, old_λ): 
				break;

			old_W = new_W
			old_λ = new_λ

			#db['compute_cost'](new_W)
			#db['compute_gradient'](new_W, new_λ)

		db['W'] = new_W
		return db['W']

	def inner_converge(self, new_W, old_W, new_λ, old_λ):
		db = self.db

		if db['convergence_method'] == 'use_W':
			diff_mag = np.linalg.norm(new_W - old_W)/np.linalg.norm(new_W)
			#print('\t\tW difference : %.8f'%diff_mag)
			if diff_mag < 0.000001:
				db['final_cost'] = db['compute_cost'](new_W)
				db['final_gradient'] = db['compute_gradient'](new_W, new_λ)	
				return True
		else:
			diff = np.linalg.norm(old_λ - new_λ)/np.linalg.norm(old_λ)
			if diff < self.conv_threshold: return True
			return False

