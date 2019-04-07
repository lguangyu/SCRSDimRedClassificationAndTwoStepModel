#!/usr/bin/env python

from .kernel_lib import *
from .Φs import *



def compute_objective_gradient(db):
	if db['kernel_type'] == 'rbf':
		return gaussian_gradient(db)
	elif db['kernel_type'] == 'rbf_slow':
		return gaussian_gradient(db)
	elif db['kernel_type'] == 'relative':
		return relative_gradient(db)
	elif db['kernel_type'] == 'linear':
		return linear_gradient(db)
	elif db['kernel_type'] == 'polynomial':
		return polynomial_gradient(db)

def linear_gradient(db):
	Φ = linear_Φ(db)
	g = 2*Φ.dot(db['W'])
	return g

def polynomial_gradient(db):
	p = db['poly_power']
	Φ = polynomial_Φ(db)
	g = (2*p*Φ).dot(db['W'])

	return g

def relative_gradient(db):
	Φ = relative_Φ(db)
	g = Φ.dot(db['W'])

	return g	



#	assumes a minimization scheme
def gaussian_gradient(db):
#	X = db['data'].X
#	N = X.shape[0]
#	d = X.shape[1]
#	σ = db['data'].σ
#
#	γ = db['compute_γ']()
#
#	[Kx, D] = Kx_D_given_W(db)
#	Ψ=γ*Kx/(σ*σ)
#
#	grad_A = np.zeros((d,d))
#	for i in range(N):
#		for j in range(N):
#			x_ij = X[i,:] - X[j,:]
#			A_ij = np.outer(x_ij, x_ij)
#
#			grad_A += Ψ[i,j] * A_ij
#
#	grad = grad_A.dot(db['W'])
#	return grad



	##	This is the faster matrix computation to check for error
	σ = db['data'].σ
	Φ = gaussian_Φ(db)
	g = Φ.dot(db['W'])
	g = 2*g/(σ*σ)

	return g	

