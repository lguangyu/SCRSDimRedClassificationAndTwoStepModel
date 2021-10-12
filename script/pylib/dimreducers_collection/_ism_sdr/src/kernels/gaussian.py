

import numpy as np
import sklearn.metrics
from ..tools import kernel_lib as klib


class gaussian():
	def __init__(self, db):
		self.db = db
		self.σ = np.median(sklearn.metrics.pairwise.pairwise_distances(db['X']))


	def __del__(self):
		pass

	def get_kernel_matrix(self, W):
		db = self.db

		X = db['X']
		σ = self.σ
		Γ = db['Γ']
	
		Kx = klib.rbk_sklearn(X.dot(W), self.σ)
		return Kx

	def get_Φ(self, W): # using the smallest eigenvalue 
		db = self.db

		X = db['X']
		σ = self.σ
		Γ = db['Γ']
	
		Kx = klib.rbk_sklearn(X.dot(W), σ)
		Ψ=Γ*Kx
		D_Ψ = klib.compute_Degree_matrix(Ψ)
		Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)	
		return Φ

	def get_Φ0(self):	# using the smallest eigenvalue 
		db = self.db 
		X = db['X']
		D_γ = klib.compute_Degree_matrix(db['Γ'])
		Φ = X.T.dot(D_γ - db['Γ']).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
		return Φ

