
import numpy as np
from classifier import *
import torch

def compare_Φ(db, Φ, Ψ):
	X = db['data'].X
	F = np.zeros((db['data'].d, db['data'].d))
	for m in range(db['data'].N):	
		for n in range(db['data'].N):	
			ΔX = X[m,:] - X[n,:]
			A_ij = np.outer(ΔX,ΔX)
			F = F + Ψ[m,n]*A_ij

	Dif = np.linalg.norm(F - Φ)
	print('Error between Φ : %.3f'%Dif)
	import pdb; pdb.set_trace()

def check_W(db, Φ, W, W_λ):
	Λ = np.diag(W_λ)
	diff = Φ.dot(W) - W.dot(Λ)
	diff_norm = np.linalg.norm(diff)
	print('\nGradient : %.3f'%diff_norm)


def initial_clustering_results(db, H):
	if 'print_debug' not in db: return
	if db['print_debug'] == False: return
	if db['data'].Y is None: return


	[allocation, db['init_spectral_nmi_on_X']] = my_spectral_clustering(db['data'].X, db['num_of_clusters'], db['data'].σ, H=H, Y=db['data'].Y)
	[allocation, db['init_kmeans_nmi_on_X']] = kmeans(db['num_of_clusters'], db['data'].X, Y=db['data'].Y)
	if 'relative_K' in db:
		db['init_nmi_relative_K'] = db['relative_K'].get_clustering_result(db['num_of_clusters'], Y=db['data'].Y)
		print('\tInitial Relative Spectral Clustering NMI on X : %.3f'%(db['init_nmi_relative_K']))

	print('\tInitial Spectral Clustering NMI on X : %.3f, σ: %.3f , σ_ratio: %.3f'%(db['init_spectral_nmi_on_X'], db['data'].σ, db["σ_ratio"]))
	print('\tInitial K-means NMI on X : %.3f'%(db['init_kmeans_nmi_on_X']))


def check_initial_knet(db):
	if 'print_debug' not in db: return
	if db['print_debug'] == False: return
	if db['data'].Y is None: return

	[AE_out, Ψx] = db['knet']( db['data'].X_Var )
	error_from_X = (db['data'].X_Var - AE_out) + (db['data'].X_Var - Ψx)
	error_from_X = torch.norm(error_from_X)
	print('\tInitial identity error is %.3f'%error_from_X.item())


def nmi_after_θ(db, network_name):	
	if 'print_debug' not in db: return
	if db['print_debug'] == False: return
	if db['data'].Y is None: return
	if network_name != 'knet': return

	[allocation, KM_nmi] = kmeans(db['num_of_clusters'], db['Ψx'], Y=db['data'].Y)

	#db['ϕ_x_normalized'] = normalize(db['Ψx'].data.numpy(), norm='l2', axis=1)
	#[allocation, KM_normalized_nmi] = kmeans(db['num_of_clusters'], db['ϕ_x_normalized'], Y=db['data'].Y)


	φ_x = ensure_matrix_is_numpy(db['Ψx'])
	DKxD = normalized_rbk_sklearn(φ_x, db['knet'].σ)
	HDKxDH = double_center(DKxD, db['H'].data.numpy())
	try:
		[U, U_normalized] = L_to_U(db, HDKxDH)
	except:
		import pdb; pdb.set_trace()

	[allocation, SP_nmi] = kmeans(db['num_of_clusters'], U, Y=db['data'].Y)
	[allocation, SP_normalized_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['data'].Y)

	print(db['current_state'])
	#print('\tKmeans NMI after θ : %.3f, Kmeans normalized after : %.3f, Spectral NMI after θ : %.3f, Spectral U_norm after : %.3f'%(KM_nmi, KM_normalized_nmi, SP_nmi, SP_normalized_nmi))
	print('\tKmeans NMI after θ : %.3f, Spectral NMI after θ on U : %.3f, Spectral U_norm after : %.3f'%(KM_nmi, SP_nmi, SP_normalized_nmi))


def nmi_after_U(db):
	if 'print_debug' not in db: return
	if db['data'].Y is None: return
	if db['print_debug'] == False: return
	if 'silent_optimization' in db: return

	[allocation, KM_nmi] = kmeans(db['num_of_clusters'], db['Ψx'], Y=db['data'].Y)
	[allocation, SP_nmi] = kmeans(db['num_of_clusters'], db['U_normalized'], Y=db['data'].Y)
	print('\tKmeans NMI after U : %.3f, Spectral NMI after U : %.3f'%(KM_nmi, SP_nmi))


def debug_print(db, msg):
	if 'print_debug' not in db: return
	if db['data'].Y is None: return
	if db['print_debug'] == False: return
	if 'silent_optimization' in db: return

	print(msg)
