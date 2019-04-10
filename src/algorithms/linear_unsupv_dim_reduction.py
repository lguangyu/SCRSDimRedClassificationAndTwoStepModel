#!/usr/bin/env python3

from algorithm import *
from kernel_lib import *
from terminal_print import *
from gradients import *
from Φs import *
from math import e
#import debug

#	max Tr[(DKuD)Kx]
#	W -> Kx -> D -> γ -> Σ ψA_i,j -> W
class linear_unsupv_dim_reduction(algorithm):
	def __init__(self, db):
		db['W'] = np.zeros((db['data'].d ,db['q']))
		#self.Ku = Ku_kernel(db['data'].Y)
		N = db['data'].N
		self.H = np.eye(N) - (1.0/N)*np.ones((N,N))

		if db['W_optimize_technique'].__name__ == 'grassman':
			db['compute_cost'] = self.grassman_cost_function
		else: db['compute_cost'] = self.compute_cost	

		db['compute_gradient'] = self.compute_f_gradient
		db['compute_Φ'] = self.compute_Φ
		db['compute_γ'] = self.update_γ	

		self.λ0 = None
		self.conv_threshold = 0.01
		self.W = None
		self.W_λ = None
		self.U_λ = None

		algorithm.__init__(self, db)
		print('Experiment : linear unsupervised dimensionality reduction\n')

	def grassman_cost_function(self, W):
		new_X = np.dot(self.db['data'].X, W)
		σ = self.db['data'].σ
		γ = self.db['compute_γ']()

		#	compute gaussian kernel
		bs = new_X.shape[0]
		K = np.empty((0, bs))	
		for i in range(bs):
			Δx = new_X[i,:] - new_X
			exp_val = -np.sum(Δx*Δx, axis=1)/(2*σ*σ)
			K = np.vstack((K, e**(exp_val)))

		return -np.sum(γ*K)

	def update_γ(self):
		db = self.db
		Ku = self.U.dot(self.U.T)

		γ = double_center(Ku, self.H)
		return γ

		#DγD = db['D_inv'].dot(γ).dot(db['D_inv'])
		#return DγD

		#return γ

	def compute_f_gradient(self, old_x):
		self.db['W'] = old_x
		return compute_objective_gradient(self.db)

	def compute_Φ(self, old_x):
		self.db['W'] = old_x

		if self.db['kernel_type'] == 'rbf':
			return gaussian_Φ(self.db)
		elif self.db['kernel_type'] == 'rbf_slow':
			return gaussian_Φ_slow(self.db)
		elif self.db['kernel_type'] == 'linear':
			return linear_Φ(self.db)
		elif self.db['kernel_type'] == 'polynomial':
			return polynomial_Φ(self.db)
		elif self.db['kernel_type'] == 'relative':
			return relative_Φ(self.db)

	def update_f(self):
		#write_to_current_line('\tAt update_f\n')
		self.db['W'] = self.optimizer.run(self.db['W'])

	def compute_cost(self, W=None):
		[Kx, D] = Kx_D_given_W(self.db, setW=W)
		γ = self.update_γ()					#DHKuHD
		return -np.sum(γ*Kx)

	def update_U(self):
		db = self.db
		k = self.db['num_of_clusters']

		[Kx, db['D_inv']] = Kx_D_given_W(db, setW=db['W'])
		L = db['D_inv'].dot(Kx).dot(db['D_inv'])
		#L = double_center(L, self.H)

		[self.U, U_λ] = eig_solver(L, k, mode='largest')

		if self.U_λ is None:
			self.U_diff = 1
		else:
			self.U_diff = np.linalg.norm(self.U_λ - U_λ)/np.linalg.norm(self.U_λ)
	
		self.U_λ = U_λ
			

	def initialize_U(self):
		db = self.db
		k = db['num_of_clusters']
		X = db['data'].X
		σ = db['data'].σ

		#L = rbk_sklearn(X, σ)       	
		#np.fill_diagonal(L, 0)

		if self.db['kernel_type'] == 'relative':
			σ_list = 1.0/relative_σ(db['data'].X)
			db['Σ'] = σ_list.dot(σ_list.T)

		[Kx, db['D_inv']] = Kx_D_given_W(db, setW=np.eye(db['data'].d))
		L = db['D_inv'].dot(Kx).dot(db['D_inv'])
		#L = double_center(L, self.H)

		[self.U, self.U_λ] = eig_solver(L, k, mode='largest')
		U_normed = normalize_U(self.U)
		[allocation, self.original_nmi] = kmeans(k, U_normed, db['data'].Y)
		print('\t\tOriginal NMI %.3f'%self.original_nmi)
		#import pdb; pdb.set_trace()	

	def initialize_W(self):
		db = self.db
		#k = db['num_of_clusters']
		#X = db['data'].X
		#Y = db['data'].Y
		#σ = db['data'].σ

		#c = 1.0/(2*σ*σ)
		#γ = self.update_γ()
		#Ψ=c*γ
		#D_Ψ = compute_Degree_matrix(Ψ)
		#Φ = 2*X.T.dot(D_Ψ - Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)



		if db['kernel_type'] == 'rbf':
			Φ = gaussian_Φ_0(db)
		elif db['kernel_type'] == 'rbf_slow':
			Φ = gaussian_Φ_0(db)
		elif db['kernel_type'] == 'linear':
			Φ = linear_Φ_0(db)
		elif db['kernel_type'] == 'polynomial':
			Φ = polynomial_Φ_0(db)
		elif db['kernel_type'] == 'relative':
			if 'Σ' not in db:
				σ_list = 1.0/relative_σ(db['data'].X)
				db['Σ'] = σ_list.dot(σ_list.T)
			Φ = relative_Φ_0(db)

		[db['W'], db['W_λ']] = eig_solver(Φ, db['q'], mode='smallest')
	

		### debug
		#nmi = self.alt_Spectral_Clustering(db)
		#print('\t\tOriginal W NMI %.3f'%nmi)
		#import pdb; pdb.set_trace()	


	def get_clustering_result(self):
		Y = self.db['data'].Y
		k = self.db['num_of_clusters']
		
		U_normed = normalize_U(self.U)
		allocation = kmeans(k, U_normed)
		return allocation

	def outer_converge(self):
		if self.U_diff < 0.01:
			#print('\tU_diff %.3f'% self.U_diff)
			return True
		else:
			#print('\tU_diff %.3f'% self.U_diff)
			return False

	def alt_Spectral_Clustering(self, db):
		k = db['num_of_clusters']
		[Kx, db['D_inv']] = Kx_D_given_W(db, setW=db['W'])
		#Kx = double_center(Kx, self.H)
		L = db['D_inv'].dot(Kx).dot(db['D_inv'])
		[self.U, self.U_λ] = eig_solver(L, k, mode='largest')
	
		U_normed = normalize_U(self.U)
		[allocation, nmi] = kmeans(k, U_normed, db['data'].Y)
		return nmi


	def verify_result(self, start_time):
		db = self.db
		if 'ignore_verification' in db: return

		k = db['num_of_clusters']
		σ = db['data'].σ

		final_cost = self.compute_cost()
		db['data'].load_validation()
		outstr = '\nExperiment : linear unsupervised dimensionality reduction : %s with final cost : %.3f\n'%(db['data_name'], final_cost)

		Y = db['data'].Y
		X = db['data'].X

		X_valid = db['data'].X_valid
		Y_valid = db['data'].Y_valid

		outstr += self.verification_basic_info(start_time)
		
		U_normed = normalize_U(self.U)
		[allocation, nmi] = kmeans(k, U_normed, Y)

		outstr += '\t\tTraining clustering NMI without dimension reduction : %.3f'%self.original_nmi + '\n'
		outstr += '\t\tTraining clustering NMI with dimension reduction : %.3f'%nmi + '\n'


		if db['separate_data_for_validation']:
			[allocation, nmi_orig] = my_spectral_clustering(X_valid, k, σ, H=self.H, Y=Y_valid)
			nmi = self.alt_Spectral_Clustering(db)

			outstr += '\t\tTest clustering NMI without dimension reduction : %.3f'%nmi_orig + '\n'
			outstr += '\t\tTest clustering NMI with dimension reduction : %.3f'%nmi + '\n'



		start_time = time.time() 
		pca = PCA(n_components=db['q'])
		X_pca1 = pca.fit_transform(X)
		Xpca = pca.transform(X_valid)
		[allocation, pca_nmi] = my_spectral_clustering(Xpca, k, σ, H=self.H, Y=Y_valid)
		pca_time = time.time() - start_time
		outstr += '\tPCA\n'
		outstr += '\t\tTraining Clustering NMI with PCA dimension reduction : %.3f'%pca_nmi + '\n'
		outstr += '\t\trun time : %.3f'%pca_time + '\n'



		print(outstr)

		fin = open('./results/LUDR_' + db['data_name'] + '_' + db['W_optimize_technique'].__name__ + '.txt', 'w') 
		fin.write(outstr)
		fin.close()

