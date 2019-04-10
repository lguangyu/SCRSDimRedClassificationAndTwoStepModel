#!/usr/bin/env python

from algorithm import *
import debug 
from RFF import *
from classifier import *
from identity_net import *
from torch.utils.data import Dataset, DataLoader
from U_optimize_cost import *
from orthogonal_optimization import *
from format_conversion import *
from basic_optimizer import *
from kernel_lib import *
from relative_kernel import *
from sn_eig_solver import *
import collections

#	max Tr[(DKuD)Kx]
#	W -> Kx -> D -> γ -> Σ ψA_i,j -> W
class sknet(algorithm):
	def __init__(self, db):
		self.db = db
		self.loss_cue = collections.deque([], 30)

		db['train_loader'] = DataLoader(dataset=db['data'], batch_size=db['batch_size'], shuffle=True, drop_last=True)
		db['iteration_count'] = 1

		#	initialize the network
		db['net_input_size'] = db['data'].d
		if(db['cuda']): db['knet'] = db['kernel_model'](db).cuda()
		else: db['knet'] = db['kernel_model'](db)
		debug.check_initial_knet(db)


		#	initialize H
		N = db['data'].N
		H = np.eye(N) - (1.0/N)*np.ones((N, N))
		db['H'] = torch.from_numpy(H)
		db['H'] = Variable(db['H'].type(db['dataType']), requires_grad=False)

		db['relative_K'] = relative_kernel(db['data'].X)

		debug.initial_clustering_results(db, H)

	def update_f(self, network_name='knet', max_epoch=4000): 		# optimize θ
		db = self.db
		#print('\t%d : Computing θ ...'%db['iteration_count'])

		if 'Ku' not in db: 
			self.update_Ku()

		HKuH = torch.mm(torch.mm(db['H'], db['Ku']), db['H'])
		DHKuHD = (HKuH*db['DD_inv']).data.numpy()
		DHKuHD = torch.from_numpy(DHKuHD)
		DHKuHD = Variable(DHKuHD.type(db['dataType']), requires_grad=False)
		db[network_name].set_Y(DHKuHD)

		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db[network_name], db, 'train_loader', loss_callback='compute_loss', epoc_loop=max_epoch)
		[db['x_hat'], db['Ψx']] = db[network_name](db['data'].X_Var)		# <- update this to be used in opt_K

		debug.nmi_after_θ(db, network_name)




	def update_U(self, network_name='knet'):
		db = self.db
		#print('\t%d : Computing U with Eigen Decomposition ...'%db['iteration_count'])
		HDKxDH = centered_normalized_rbk_sklearn(db['Ψx'].data.numpy(), db['data'].σ, db['H'].data.numpy())
		#DKxD = centered_normalized_rbk_sklearn(db['Ψx'].data.numpy(), db['data'].σ, None)
		HDKxDH = torch.from_numpy(HDKxDH)
		HDKxDH = Variable(HDKxDH.type(db['dataType']), requires_grad=False)

		[db['U'], db['U_normalized']] = db['sn_eig_solver'].obtain_eigen_vectors(HDKxDH, db['data'].X_Var)
		#db['U'] = db['U_normalized']	# reset U to the solution of spectral clustering			<------------------------------ remember to delete
		
		db['prev_Ku'] = db['Ku']
		self.update_Ku()
		db['iteration_count'] += 1
		#db['l1_ratio'] = 0.5*torch.abs(db['obj_loss']/db['l1_loss']).item()

		debug.nmi_after_U(db)

	def update_Ku(self):
		db = self.db

		Ku = db['U'].dot(db['U'].T)
		db['Ku'] = torch.from_numpy(Ku)
		db['Ku'] = Variable(db['Ku'].type(db['dataType']), requires_grad=False)

		return [db['Ku'], Ku]

	def initialize_U(self):
		db = self.db

		db['learning_rate'] = 0.001
		db['add_decoder'] = False
		db['mlp_width'] = 200
		db['net_depth'] = 3
		db['sn_eig_solver'] = sn_eig_solver(db)

		[AE_out, Ψx] = db['knet'](db['data'].X_Var)

		DKxD = db['relative_K'].get_kernel(center_kernel=False)

		DD_inv = db['relative_K'].Dv
		DD_inv = torch.from_numpy(DD_inv)
		db['DD_inv'] = Variable(DD_inv.type(db['dataType']), requires_grad=False)
		[db['U'], db['U_normalized']] = db['sn_eig_solver'].obtain_eigen_vectors(DKxD, db['data'].X_Var)
		[Ku_torch, Ku_numpy] = self.update_Ku()

		[allocation, db['init_spectral_nmi_on_Ψx']] = kmeans(db['num_of_clusters'], db['U'], Y=db['data'].Y)
		[allocation, db['init_spectral_nmi_on_Ψx_U_norm']] = kmeans(db['num_of_clusters'], db['U_normalized'], Y=db['data'].Y)
		[allocation, db['init_kmeans_nmi_on_Ψx']] = kmeans(db['num_of_clusters'], Ψx, Y=db['data'].Y)

		#db['init_HSIC'] = np.sum(Ku_numpy*HDKxDH)
		

		#print('\tInitial HSIC %.3f'%(db['init_HSIC']))
		print('\tInitial Spectral Clustering NMI on RFF Ψx : %.3f, σ: %.3f , σ_ratio: %.3f'%(db['init_spectral_nmi_on_Ψx'], db['data'].σ, db["σ_ratio"]))
		print('\tInitial Spectral Clustering NMI on RFF Ψx U normalized : %.3f'%(db['init_spectral_nmi_on_Ψx_U_norm']))
		print('\tInitial K-means NMI on Ψx : %.3f'%(db['init_kmeans_nmi_on_Ψx']))

		#db['U'] = db['U_normalized']	# initialize U to the solution of spectral clustering		<------------------------------ remember to delete

	def initialize_W(self):
		self.db['knet'].learning_rate = 0.001
		return


		initializing_W = True

		if initializing_W:	
			db = self.db
			db['silent_optimization'] = True
			lr = 1
			print('\tPicking the appropriate learning rate...')
	
			while True:
				if(db['cuda']): db['tmp_model'] = db['kernel_model'](db, learning_rate=lr, silent=True).cuda()
				else: db['tmp_model'] = db['kernel_model'](db, learning_rate=lr, silent=True)
		
				l1 = []
				l2 = []
				l3 = []
				l4 = []
				l5 = []
				print('\tTesting learning rate :%.6f'%lr)

				self.update_f(network_name='tmp_model', max_epoch=4)
				print('\t\tLoss before step : %.4f , loss after step : %.4f'%(db['loss_begin'],db['loss_end']))

				if db['loss_begin'] <= db['loss_end'] and (db['loss_end'] - db['loss_begin']) > (0.001*np.absolute(db['loss_begin'])):
					lr = lr/2.0
					continue
				else:
					break

			print('\tUsing learning rate of %.8f'%lr)
			self.db['knet'].learning_rate = lr
			db.pop('silent_optimization')


	def one_more_run(self):
		db = self.db

		allocation = KMeans(db['num_of_clusters'], n_init=10).fit_predict(db['U'])
		db['U'] = Allocation_2_Y(allocation)
		db['λ'] = 0 
		db['l1_ratio'] = 0 

		self.update_Ku()
		self.update_f()
		self.update_U()


	def outer_converge(self):
		db = self.db
		self.loss_cue.append(db['loss'])
		slope_exit = 0.0005

		Ku_diff = torch.norm(db['prev_Ku'] - db['Ku']).item()
		Ku_size = torch.norm(db['prev_Ku']).item()
		change_size = Ku_diff/Ku_size
		debug.debug_print(db, '\t\t------ Ku_diff : %.5f, percent change : %.5f'%(Ku_diff, change_size))

		if len(self.loss_cue) < 3: return False
		progression_slope = get_slope(self.loss_cue)
		debug.debug_print(db, '\t\t------ Progress slope : %.5f, exit when slope < %.9f'%(progression_slope,slope_exit))

		if db['iteration_count'] > 5: 
			print('a')
			return True
		if change_size < 0.01: 
			print('b')
			return True
		if np.absolute(progression_slope) < slope_exit: 
			print('c')
			self.one_more_run()
			return True
		return False


	def verify_result(self, start_time):
		pass

