

from sp_net import *
from specNet_optimizer import *
from basic_optimizer import *

class sn_eig_solver():
	def __init__(self, db):
		self.spn = sp_net(db)
		self.db = db

	def L_to_U(self, L, k, return_eig_val=False):
		L = ensure_matrix_is_numpy(L)
		eigenValues,eigenVectors = np.linalg.eigh(L)
	
		n2 = len(eigenValues)
		n1 = n2 - k
		U = eigenVectors[:, n1:n2]
		U_lambda = eigenValues[n1:n2]
		U_normalized = normalize(U, norm='l2', axis=1)
		
		if return_eig_val: return [U, U_normalized, U_lambda]
		else: return [U, U_normalized]

	def obtain_eigen_vectors(self, K, X, Dloader_name='train_loader'):
		db = self.db

		if K.shape[0] < 1200:
			[U, U_normalized] = self.L_to_U(K, db['num_of_clusters'])
		else:
			K = torch.from_numpy(K)
			K = Variable(DKxD.type(db['dataType']), requires_grad=False)

			self.spn.set_Laplacian(K)
			m_sqrt = np.sqrt(X.shape[0])		
		
			basic_optimizer(self.spn, db, Dloader_name)
			Y = self.spn.get_orthogonal_out(X)/m_sqrt	# Y^TY = I
			U = Y.data.numpy()
			U_normalized = normalize(U, norm='l2', axis=1)

		return [U, U_normalized]

