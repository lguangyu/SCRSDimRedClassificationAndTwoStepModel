#!/usr/bin/env python

from .kernel_lib import *


def rank_constraint(db):
	A = db['W'].dot(db['W'].T) + 0.001*np.eye(db['data'].d)
	return np.linalg.inv(A)


def relative_Φ_0(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()
	Ψ = db['Σ']*γ

	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)

	if db['λ_ratio'] != 0: 
		λ = np.linalg.norm(Φ)/np.linalg.norm(rank_constraint(db))
		Φ = Φ + λ*rank_constraint(db) 	# using rank constraint 

	return Φ

def gaussian_Φ_0(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()
	D_γ = compute_Degree_matrix(γ)

	Φ = X.T.dot(D_γ - γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)

	if db['λ_ratio'] != 0: 
		λ = np.linalg.norm(Φ)/np.linalg.norm(rank_constraint(db))
		Φ = Φ + λ*rank_constraint(db) 	# using rank constraint 

	return Φ

def polynomial_Φ_0(db):
	X = db['data'].X
	p = db['poly_power']
	c = db['poly_constant']

	γ = db['compute_γ']()
	Φ = -X.T.dot(γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)

	if db['λ_ratio'] != 0: 
		λ = np.linalg.norm(Φ)/np.linalg.norm(rank_constraint(db))
		Φ = Φ + λ*rank_constraint(db) 	# using rank constraint 

	return Φ

def linear_Φ_0(db):
	X = db['data'].X
	γ = db['compute_γ']()
	Φ = -X.T.dot(γ).dot(X); 			

	if db['λ_ratio'] != 0: 
		λ = np.linalg.norm(Φ)/np.linalg.norm(rank_constraint(db))
		Φ = Φ + λ*rank_constraint(db) 	# using rank constraint 

	return Φ

def gaussian_Φ_slow(db):
	X = db['data'].X
	σ = db['data'].σ
	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx

	Φ = np.zeros((db['data'].d, db['data'].d))
	for m in range(db['data'].N):	
		for n in range(db['data'].N):	
			ΔX = X[m,:] - X[n,:]
			A_ij = np.outer(ΔX,ΔX)
			Φ = Φ + Ψ[m,n]*A_ij

	if db['λ_ratio'] != 0: Φ = Φ + db['λ']*rank_constraint(db) 	# using rank constraint 

	return Φ

def gaussian_Φ(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx
	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)

	if db['λ_ratio'] != 0: Φ = Φ + db['λ']*rank_constraint(db) 	# using rank constraint 
	return Φ

def relative_Φ(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=db['Σ']*γ*Kx
	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)

	if db['λ_ratio'] != 0: Φ = Φ + db['λ']*rank_constraint(db) 	# using rank constraint 
	return Φ

def polynomial_Φ(db):
	X = db['data'].X
	p = db['poly_power']
	c = db['poly_constant']

	Kx = poly_sklearn(X, p-1, c)
	γ = db['compute_γ']()
	Ψ = γ*Kx
	Φ = -X.T.dot(Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)

	if db['λ_ratio'] != 0: Φ = Φ + db['λ']*rank_constraint(db) 	# using rank constraint 
	return Φ

def linear_Φ(db):
	X = db['data'].X
	γ = db['compute_γ']()
	Φ = -X.T.dot(γ).dot(X); 			

	if db['λ_ratio'] != 0: Φ = Φ + db['λ']*rank_constraint(db) 	# using rank constraint 
	return Φ
