#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import sys
#sys.path.append('./src')
#sys.path.append('./tests')
#sys.path.append('./src/data_loader')
#sys.path.append('./src/helper')
#sys.path.append('./src/algorithms')
#sys.path.append('./src/optimization')
#sys.path.append('./tests/linear_supervised_dim_reduction')
#sys.path.append('./tests/linear_unsupervised_dim_reduction')
import numpy as np
from .algorithms.linear_supv_dim_reduction import *
from .algorithms.linear_unsupv_dim_reduction import *
from .hsic_parent import *
from .data_loader.np_loader import *
from .optimization.ism import *
from .optimization.orthogonal_optimization import *
from .optimization.DimGrowth import *
import time 

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class hsic_algorithms(hsic_parent):
	def __init__(self, db):
		hsic_parent.__init__(self, db)

	def run(self):
		db = self.db
		db['Dloader'] = db['data_loader'](db)
		TF = db['TF_obj'](db)
		TF.initialize_U()
		TF.initialize_W()

		start_time = time.time() 
		while True:
			TF.update_f()
			TF.update_U()
			if TF.outer_converge(): break;

		TF.verify_result(start_time)
		

if __name__ == "__main__":
	#print(sys.argv[1], sys.argv[2])

	db = {}
	fin = open(sys.argv[1],'r')
	cmds = fin.readlines()
	fin.close()

	for i in cmds: exec(i)

	hs = hsic_algorithms(db)
	hs.run()
