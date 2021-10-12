

import numpy as np
import sklearn.metrics
from ..tools import kernel_lib as klib


class squared():
	def __init__(self, db):
		self.db = db

	def __del__(self):
		pass

	def get_kernel_matrix(self, W):
		db = self.db

		X = db['X'].dot(W)
		bs = X.shape[0]
		K = np.zeros((bs,bs))

		for i in range(bs):
			dif = X[i,:] - X
			K[i,:] = np.sum(dif*dif, axis=1)

		return -K


	def get_Φ(self, W): # using the smallest eigenvalue 
		db = self.db 
		X = db['X']

		D_γ = klib.compute_Degree_matrix(db['Γ'])
		Φ = X.T.dot(D_γ - db['Γ']).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
		return Φ

	def get_Φ0(self):	# using the smallest eigenvalue 
		return self.get_Φ(None)



