#!/usr/bin/env python3

import sklearn.metrics
import autograd.numpy as np
import torch
from format_conversion import *
from sklearn.preprocessing import normalize			# version : 0.17
from sklearn.neighbors import NearestNeighbors

def Y_2_allocation(Y):
	i = 0
	allocation = np.array([])
	for m in range(Y.shape[0]):
		allocation = np.hstack((allocation, np.where(Y[m] == 1)[0][0]))
		i += 1

	return allocation


def Allocation_2_Y(allocation):
	
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m)[0]
		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,num_of_classes))
	Y = np.delete(Y, 0, 0)

	return Y

def Kx_D_given_W(db, setX=None, setW=None):
	if setX is None: outX = db['data'].X.dot(db['W'])
	else: outX = setX.dot(db['W'])
	
	if setW is None: outX = db['data'].X.dot(db['W'])
	else: outX = db['data'].X.dot(setW)

	if db['kernel_type'] == 'rbf':
		Kx = rbk_sklearn(outX, db['data'].σ)
	elif db['kernel_type'] == 'relative':
		Kx = rbk_relative_σ(db, outX)
	elif db['kernel_type'] == 'rbf_slow':
		Kx = rbk_sklearn(outX, db['data'].σ)
	elif db['kernel_type'] == 'linear':
		Kx = outX.dot(outX.T)
	elif db['kernel_type'] == 'polynomial':
		Kx = poly_sklearn(outX, db['poly_power'], db['poly_constant'])


	np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	D = compute_inverted_Degree_matrix(Kx)
	return [Kx, D]


def poly_sklearn(data, p, c):
	poly = sklearn.metrics.pairwise.polynomial_kernel(data, degree=p, coef0=c)
	return poly

def centered_normalized_rbk_sklearn(X, σ, H):
	Kx = rbk_sklearn(X, σ)       	
	Dinv = 1.0/np.sqrt(Kx.sum(axis=1))
	Dv = np.outer(Dinv,Dinv)
	if H is None:
		return Dv*Kx
	else:
		return H.dot(Dv*Kx).dot(H)

def normalized_rbk_sklearn(X, σ):
	Kx = rbk_sklearn(X, σ)       	
	D = compute_inverted_Degree_matrix(Kx)
	return D.dot(Kx).dot(D)

def rbk_sklearn(data, sigma):
	gammaV = 1.0/(2*sigma*sigma)
	rbk = sklearn.metrics.pairwise.rbf_kernel(data, gamma=gammaV)
	np.fill_diagonal(rbk, 0)			#	Set diagonal of adjacency matrix to 0
	return rbk


def rbk_relative_σ(db, X, Y=None):	#This should take 2 values
	D = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean', n_jobs=1)
	K = np.exp(-(D*D*db['Σ'])/2.0)
	np.fill_diagonal(K,0)
	return K

def get_RFF_embeding(db, X):
	[HDKDH, AE_out, Ψx] = get_RFF_kernel(db, X)
	L = HDKDH.data.numpy()

	[U, U_normalized] = L_to_U(db, L)
	return [U, U_normalized, L]

def get_RFF_raw_kernel(db, Ψx):
	K = db['RFF'].get_rbf(Ψx, db['data'].σ, output_torch=True, dtype=db['dataType'])
	return K

def get_RFF_kernel(db, X, network_name='knet'):
	[AE_out, Ψx] = db[network_name]( X )
	
	K = db['RFF'].get_rbf(Ψx, db['data'].σ, output_torch=True, dtype=db['dataType'])
	D = 1/torch.sqrt(torch.sum(K, dim=0))
	D2 = torch.ger(D,D)
	db['DD_inv'] = D2
	K = K*D2

	HDKDH = torch.mm(torch.mm(db['H'], K), db['H'])
	return [HDKDH, AE_out, Ψx]
	


def Ku_kernel(labels):
	Y = Allocation_2_Y(labels)
	Ky = Y.dot(Y.T)
	
	return Ky

def double_center(M, H):
	M = ensure_matrix_is_numpy(M)
	H = ensure_matrix_is_numpy(H)
	HMH = H.dot(M).dot(H)
	return HMH

def L_to_U(db, L, return_eig_val=False):
	L = ensure_matrix_is_numpy(L)
	eigenValues,eigenVectors = np.linalg.eigh(L)

	n2 = len(eigenValues)
	n1 = n2 - db['num_of_clusters']
	U = eigenVectors[:, n1:n2]
	U_lambda = eigenValues[n1:n2]
	U_normalized = normalize(U, norm='l2', axis=1)
	
	if return_eig_val: return [U, U_normalized, U_lambda]
	else: return [U, U_normalized]

def nomalized_by_Degree_matrix(M, D):
	D2 = np.diag(D)
	DMD = M*(np.outer(D2, D2))
	return DMD

def compute_inverted_Degree_matrix(M):
	return np.diag(1.0/np.sqrt(M.sum(axis=1)))

def compute_Degree_matrix(M):
	return np.diag(np.sum(M, axis=0))


def normalize_U(U):
	return normalize(U, norm='l2', axis=1)


def eig_solver(L, k, mode='smallest'):
	#L = ensure_matrix_is_numpy(L)
	eigenValues,eigenVectors = np.linalg.eigh(L)

	#print(eigenValues < 0)
	#import pdb; pdb.set_trace()

	if mode == 'smallest':
		U = eigenVectors[:, 0:k]
		U_λ = eigenValues[0:k]
	elif mode == 'largest':
		n2 = len(eigenValues)
		n1 = n2 - k
		U = eigenVectors[:, n1:n2]
		U_λ = eigenValues[n1:n2]
	else:
		raise ValueError('unrecognized mode : ' + str(mode) + ' found.')
	
	return [U, U_λ]


def relative_σ(X):
	n = X.shape[0]
	if n < 50: num_of_samples = n
	else: num_of_samples = 50
	
	unique_X = np.unique(X, axis=0)
	neigh = NearestNeighbors(num_of_samples)

	neigh.fit(unique_X)
	
	[dis, idx] = neigh.kneighbors(X, num_of_samples, return_distance=True)
	dis_inv = 1/dis[:,1:]
	idx = idx[:,1:]
	
	total_dis = np.sum(dis_inv, axis=1)
	total_dis = np.reshape(total_dis,(n, 1))
	total_dis = np.matlib.repmat(total_dis, 1, num_of_samples-1)
	dis_ratios = dis_inv/total_dis

	result_store_dictionary = {}
	σ_list = np.zeros((n,1))
	
	for i in range(n):
		if str(X[i,:]) in result_store_dictionary:
			σ = result_store_dictionary[str(X[i,:])] 
			σ_list[i] = σ
			continue

		dr = dis_ratios[i,:]

		Δ = unique_X[idx[i,:],:] - X[i,:]
		Δ2 = Δ*Δ
		d = np.sum(Δ2,axis=1)
		σ = np.sqrt(np.sum(dr*d))
		σ_list[i] = σ#*10

		result_store_dictionary[str(X[i,:])] = σ

	#return σ_list.dot(σ_list.T)
	return σ_list

	

