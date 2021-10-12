

import numpy as np
import sklearn.metrics
from ..tools import kernel_lib as klib


class linear():
	def __init__(self, db):
		self.db = db

	def __del__(self):
		pass

	def get_kernel_matrix(self, W):
		db = self.db

		X = db['X'].dot(W)
		return X.dot(X.T)


	def get_Φ(self, W): # using the smallest eigenvalue 
		db = self.db 
		X = db['X']

		Φ = -X.T.dot(db['Γ']).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
		return Φ

	def get_Φ0(self):	# using the smallest eigenvalue 
		return self.get_Φ(None)



