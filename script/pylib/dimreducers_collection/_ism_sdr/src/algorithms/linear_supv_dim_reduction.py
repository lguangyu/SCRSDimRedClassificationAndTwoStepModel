#!/usr/bin/env python3

from .algorithm import *
from ..tools import kernel_lib as klib

#import sys
#import debug

#	max Tr[(DKuD)Kx]
#	W -> Kx -> D -> γ -> Σ ψA_i,j -> W
class linear_supv_dim_reduction(algorithm):
	def __init__(self, db):
		self.db = db
		db['Γ'] = db['H'].dot(db['Y']).dot(db['Y'].T).dot(db['H'])
		db['compute_cost'] = self.compute_HSIC
		db['compute_gradient'] = self.compute_gradient


	def __del__(self):
		pass	

	def initialize_U(self):
		pass

	def initialize_W(self):
		db = self.db
		db['init_HSIC'] = self.compute_HSIC(np.eye(db['d']))
	
		Φ0 = self.db['kernel'].get_Φ0()
		[db['W'], eigs] = klib.eig_solver(Φ0, db['q'])

	def update_U(self):
		pass

	def update_f(self):
		self.db['W'] = self.db['optimizer'].run(self.db['W'])
		self.db['final_HSIC'] = self.compute_HSIC(self.db['W'])

	def outer_converge(self):
		return True

	def verify_result(self, start_time):
		pass


	def compute_HSIC(self, W):
		db = self.db

		Kx = db['kernel'].get_kernel_matrix(W)
		HSIC_val = np.sum(db['Γ']*Kx)
		return HSIC_val

	def compute_gradient(self, W, Λ, Φ=None):
		db = self.db 

		σ = db['kernel'].σ
		if Φ is None: Φ = db['kernel'].get_Φ(W)
		gradient = Φ.dot(W) - W.dot(np.diag(Λ))
		return gradient			


