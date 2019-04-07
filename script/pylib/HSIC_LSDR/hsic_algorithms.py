#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
from .algorithms.linear_supv_dim_reduction import *
from .algorithms.linear_unsupv_dim_reduction import *
from .hsic_parent import *
from .data_loader.basic_dataset import *
from .algorithms.knet import *
from .algorithms.sknet import *
from .optimization.ism import *
from .optimization.orthogonal_optimization import *
from .optimization.DimGrowth import *
from .optimization.grassman import *
import time 

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class hsic_algorithms(hsic_parent):
	def __init__(self, db):
		hsic_parent.__init__(self, db)
		if 'cuda' in db:
			if(db['cuda']): db['dataType'] = torch.cuda.FloatTensor
			else: db['dataType'] = torch.FloatTensor				

	def run(self):
		db = self.db
		db['data'] = db['dataset_class'](db)
		self.TF = db['TF_obj'](db)
		self.TF.initialize_U()
		self.TF.initialize_W()

		start_time = time.time() 
		while True:
			self.TF.update_f()
			self.TF.update_U()
			if self.TF.outer_converge(): break;

		self.TF.verify_result(start_time)
		

if __name__ == "__main__":
	#print(sys.argv[1], sys.argv[2])

	db = {}
	fin = open(sys.argv[1],'r')
	cmds = fin.readlines()
	fin.close()

	for i in cmds: exec(i)

	hs = hsic_algorithms(db)
	hs.run()
