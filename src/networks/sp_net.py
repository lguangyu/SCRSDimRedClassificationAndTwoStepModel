#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time 

class sp_net(torch.nn.Module):
	def __init__(self, db):
		super(sp_net, self).__init__()
		self.db = db
		self.learning_rate = db['learning_rate']
		self.input_size = db['net_input_size']
		self.output_dim = db['num_of_clusters']
		self.mlp_width = db['mlp_width']
		self.net_depth = db['net_depth']
		self.dataType = db['dataType']
		self.add_decoder = db['add_decoder']


		in_out_list = []
		for l in range(1, self.net_depth+1):
			if l == self.net_depth:
				in_out_list.append((self.output_dim ,self.mlp_width))
				lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(self.mlp_width) + ', ' + str(self.output_dim) + ' , bias=True)'
				exec(lr)
				exec('self.l' + str(l) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none
			elif l == 1:
				in_out_list.append((self.mlp_width ,self.input_size))
				lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(self.input_size) + ', ' + str(self.mlp_width) + ' , bias=True)'
				exec(lr)
				exec('self.l' + str(l) + '.activation = "relu"')		#softmax, relu, tanh, sigmoid, none
			else:
				in_out_list.append((self.mlp_width, self.mlp_width))
				lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(self.mlp_width) + ', ' + str(self.mlp_width) + ' , bias=True)'
				exec(lr)
				exec('self.l' + str(l) + '.activation = "relu"')


		if db['add_decoder']:
			in_out_list.reverse()
	
			for l, item in enumerate(in_out_list):
				c = l + self.net_depth + 1
				if c == self.net_depth*2:
					lr = 'self.l' + str(c) + ' = torch.nn.Linear(' + str(item[0]) + ', ' + str(item[1]) + ' , bias=True)'
					exec(lr)
					exec('self.l' + str(c) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none
				else:
					lr = 'self.l' + str(c) + ' = torch.nn.Linear(' + str(item[0]) + ', ' + str(item[1]) + ' , bias=True)'
					exec(lr)
					exec('self.l' + str(c) + '.activation = "relu"')


		self.initialize_network()
		self.output_network()

	def output_network(self):
		print('\tConstructing Kernel Net')
		for i in self.children():
			try:
				print('\t\t%s , %s'%(i,i.activation))
			except:
				print('\t\t%s '%(i))

	def set_Laplacian(self, Laplacian):
		self.Laplacian = Laplacian

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

	def initialize_network(self):
		db = self.db

		for param in self.parameters():
			if(len(param.data.numpy().shape)) > 1:
				torch.nn.init.kaiming_normal_(param.data , a=0, mode='fan_in')	
			else:
				pass
				#param.data = torch.zeros(param.data.size())

		self.num_of_linear_layers = 0
		for m in self.children():
			if type(m) == torch.nn.Linear:
				self.num_of_linear_layers += 1

	def get_orthogonal_out(self, x):
		m_sqrt = np.sqrt(x.shape[0])		
		y = self.forward(x)

		rk = torch.matrix_rank(y).item()
		if rk < self.db['num_of_clusters']:
			import pdb; pdb.set_trace()

		YY = torch.mm(torch.t(y), y)
		L = torch.cholesky(YY)

		Li = m_sqrt*torch.t(torch.inverse(L))
		Yout = torch.mm(y, Li)
		return Yout

	def compute_loss(self, x, labels, indices):
		Yout = self.get_orthogonal_out(x)
		K = self.distance_kernel(Yout)

		PP = self.Laplacian[indices, :]
		Ksmall = PP[:, indices]
		obj_loss = torch.sum(K*Ksmall)

		return obj_loss


	def distance_kernel(self, x):			#Each row is a sample
		bs = x.shape[0]
		K = self.db['dataType'](bs, bs)
		K = Variable(K.type(self.db['dataType']), requires_grad=False)		

		for i in range(bs):
			dif = x[i,:] - x
			K[i,:] = torch.sum(dif*dif, dim=1)

		return K


	def gaussian_kernel(self, x, σ):			#Each row is a sample
		bs = x.shape[0]
		K = self.db['dataType'](bs, bs)
		K = Variable(K.type(self.db['dataType']), requires_grad=False)		

		for i in range(bs):
			dif = x[i,:] - x
			K[i,:] = torch.exp(-torch.sum(dif*dif, dim=1)/(2*σ*σ))

		return K

	def forward(self, y0):
		for m, layer in enumerate(self.children(),1):
			if m == self.net_depth*2:
				cmd = 'self.y_pred = self.l' + str(m) + '(y' + str(m-1) + ')'
				#print(cmd)	
				exec(cmd)
				break;
			elif m == self.net_depth:
				if self.add_decoder:
					var = 'y' + str(m)
					cmd = var + ' = self.l' + str(m) + '(y' + str(m-1) + ')'
					#print(cmd)	
					exec(cmd)
				else:
					cmd = 'self.y_pred = self.l' + str(m) + '(y' + str(m-1) + ')'
					#print(cmd)	
					exec(cmd)
					return self.y_pred

			else:
				var = 'y' + str(m)
				cmd = var + ' = F.relu(self.l' + str(m) + '(y' + str(m-1) + '))'
				#print(cmd)	
				exec(cmd)
				#exec(cmd2)

		exec('self.fx = y' + str(self.net_depth))
		return [self.y_pred, self.fx]

