

import numpy as np
import sklearn.metrics
from ..tools import kernel_lib as klib


class polynomial():
	def __init__(self, db):
		self.db = db
		if 'polynomial_degree' in db: self.p = db['polynomial_degree']
		else: self.p = 3

		if 'poly_constant' in db: self.c = db['poly_constant']
		else: self.c = 1


	def __del__(self):
		pass

	def get_kernel_matrix(self, W):
		db = self.db

		X = db['X']
		p = self.p
		c = self.c
	
		Kx = klib.poly_sklearn(X.dot(W), p-1, c)
		return Kx

	def get_Φ(self, W): # using the smallest eigenvalue 
		db = self.db

		X = db['X']
		p = self.p
		c = self.c
	
		Kx = klib.poly_sklearn(X.dot(W), p-1, c)
		Ψ = db['Γ']*Kx
		Φ = -X.T.dot(Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
		return Φ


	def get_Φ0(self):	# using the smallest eigenvalue 
		db = self.db 
		X = db['X']
		Φ = -X.T.dot(db['Γ']).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
		return Φ


