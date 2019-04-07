#!/usr/bin/env python3


from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import KFold
import numpy as np
from subprocess import call
from .data_loader.basic_dataset import *
from .helper.path_tools import *
from .optimization.ism import *
from .optimization.orthogonal_optimization import *
from .optimization.DimGrowth import *
import itertools

#from acc import *
import socket
import types
import torch
import pickle
import random
import string
import os

class test_base():
	def __init__(self, db):
		db['data_file_name'] = './datasets/' + db['data_name'] + '.csv'
		db['label_file_name'] = './datasets/' + db['data_name'] + '_label.csv'


		if db['separate_data_for_validation']:
			db['validation_data_file_name'] = './datasets/' + db['data_name'] + '_validation.csv'
			db['validation_label_file_name'] = './datasets/' + db['data_name'] + '_label_validation.csv'
		else:
			db['validation_data_file_name'] = './datasets/' + db['data_name'] + '.csv'
			db['validation_label_file_name'] = './datasets/' + db['data_name'] + '_label.csv'



		db['best_path'] = '../version9/pre_trained_weights/Best_pk/' 
		db['center_and_scale'] = True
		db['poly_power'] = 3
		db['poly_constant'] = 1
		if 'kernel_type' not in db: db['kernel_type'] = 'relative'		#rbf, linear, rbf_slow, polynomial, relative
		self.db = db


		tmp_path = './tmp/' + db['data_name'] + '/'
		db_output_path = tmp_path + 'db_files/'
		batch_output_path = tmp_path + 'batch_outputs/'

		ensure_path_exists('./tmp')
		ensure_path_exists('./results')
		ensure_path_exists(tmp_path)
		ensure_path_exists(db_output_path)
		ensure_path_exists(batch_output_path)




	def remove_tmp_files(self):
		db = self.db
		file_in_tmp = os.listdir('./tmp/' + db['data_name'] + '/db_files/')
		for i in file_in_tmp:
			if i.find(db['data_name']) == 0:
				os.remove('./tmp/' + db['data_name'] + '/db_files/' + i)


	def output_db_to_text(self):
		db = self.db
		db['db_file']  = './tmp/' + db['data_name'] + '/db_files/' + db['data_name'] + '_' +  str(int(10000*np.random.rand())) + '.txt'
		fin = open(db['db_file'], 'w')

		for i,j in db.items():
			if type(j) == str:
				fin.write('db["' + i + '"]="' + str(j) + '"\n')
			elif type(j) == bool:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == type:
				fin.write('db["' + i + '"]=' + j.__name__ + '\n')
			elif type(j) == float:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == int:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == types.FunctionType:
				fin.write('db["' + i + '"]=' + j.__name__ + '\n')
			elif j is None:
				fin.write('db["' + i + '"]=None\n')
			else:
				print(i,j)
				raise ValueError('unrecognized type : ' + str(type(j)) + ' found.')

		fin.close()
		return db['db_file']


	def export_bash_file(self, i, test_name, export_db):
		run_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(2))

		cmd = ''
		cmd += "#!/bin/bash\n"
		cmd += "\n#set a job name  "
		cmd += "\n#SBATCH --job-name=%d_%s_%s"%(i, test_name, run_name)
		cmd += "\n#################  "
		cmd += "\n#a file for job output, you can check job progress"
		cmd += "\n#SBATCH --output=./tmp/%s/batch_outputs/%d_%s_%s.out"%(test_name, i, test_name, run_name)
		cmd += "\n#################"
		cmd += "\n# a file for errors from the job"
		cmd += "\n#SBATCH --error=./tmp/%s/batch_outputs/%d_%s_%s.err"%(test_name, i, test_name, run_name)
		cmd += "\n#################"
		cmd += "\n#time you think you need; default is one day"
		cmd += "\n#in minutes in this case, hh:mm:ss"
		#cmd += "\n#SBATCH --time=24:00:00"
		cmd += "\n#################"
		cmd += "\n#number of tasks you are requesting"
		cmd += "\n#SBATCH -N 1"
		cmd += "\n#SBATCH --exclusive"
		cmd += "\n#################"
		cmd += "\n#partition to use"
		#cmd += "\n#SBATCH --partition=general"
		cmd += "\n#SBATCH --partition=ioannidis"	
		#cmd += "\n#SBATCH --partition=gpu"	
		cmd += "\n#SBATCH --constraint=E5-2680v2@2.80GHz"		# 20 cores	
		#cmd += "\n#SBATCH --exclude=c3096"
		cmd += "\n#SBATCH --mem=120Gb"
		cmd += "\n#################"
		cmd += "\n#number of nodes to distribute n tasks across"
		cmd += "\n#################"
		cmd += "\n"
		cmd += "\npython ./src/hsic_algorithms.py " + export_db
		
		fin = open('execute_combined.bash','w')
		fin.write(cmd)
		fin.close()

#	def batch_run_10_fold(self):
#		count = 0
#		db = self.db
#		output_list = self.parameter_ranges()
#		every_combination = list(itertools.product(*output_list))
#		
#		for count, single_instance in enumerate(every_combination):
#			[W_optimize_technique, repeat_run] = single_instance
#
#			db['W_optimize_technique'] = W_optimize_technique
#			fname = self.output_db_to_text()
#			self.export_bash_file(count, db['data_name'], fname)
#
#			if socket.gethostname().find('login') != -1:
#				call(["sbatch", "execute_combined.bash"])
#			else:
#				os.system("bash ./execute_combined.bash")


	def batch_run(self):
		count = 0
		db = self.db
		output_list = self.parameter_ranges()
		every_combination = list(itertools.product(*output_list))
		
		for count, single_instance in enumerate(every_combination):
			[W_optimize_technique, repeat_run] = single_instance

			db['W_optimize_technique'] = W_optimize_technique
			fname = self.output_db_to_text()
			self.export_bash_file(count, db['data_name'], fname)

			if socket.gethostname().find('login') != -1:
				call(["sbatch", "execute_combined.bash"])
			else:
				os.system("bash ./execute_combined.bash")


	def batch_file_names(self):
		count = 0
		db = self.db
		output_list = self.file_name_ranges()
		every_combination = list(itertools.product(*output_list))

		for count, single_instance in enumerate(every_combination):
			[data_name, W_optimize_technique] = single_instance
			db['data_name'] = data_name
			db['W_optimize_technique'] = W_optimize_technique
			
			tmp_path = './tmp/' + db['data_name'] + '/'
			db_output_path = tmp_path + 'db_files/'
			batch_output_path = tmp_path + 'batch_outputs/'

			ensure_path_exists('./tmp')
			ensure_path_exists(tmp_path)
			ensure_path_exists(db_output_path)
			ensure_path_exists(batch_output_path)

			fname = self.output_db_to_text()
			self.export_bash_file(count, db['data_name'], fname)

			if socket.gethostname().find('login') != -1:
				call(["sbatch", "execute_combined.bash"])
			else:
				os.system("bash ./execute_combined.bash")


	def basic_run(self):
		self.remove_tmp_files()
		fname = self.output_db_to_text()

		call(["./src/hsic_algorithms.py", fname])

	def kick_off_single_from_10_fold(self, indx):
		db = self.db
		original_name = db['data_name']

		db['data_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(indx) + '.csv'
		db['label_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(indx) + '_label.csv'

		db['validation_data_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(indx) + '_validation.csv'
		db['validation_label_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(indx) + '_label_validation.csv'

		db['data_name'] = original_name + '_' + str(indx)

		tmp_path = './tmp/' + db['data_name'] + '/'
		db_output_path = tmp_path + 'db_files/'
		batch_output_path = tmp_path + 'batch_outputs/'
		ensure_path_exists('./tmp')
		ensure_path_exists('./results')
		ensure_path_exists(tmp_path)
		ensure_path_exists(db_output_path)
		ensure_path_exists(batch_output_path)

		fname = self.output_db_to_text()
		call(["./src/hsic_algorithms.py", fname])


	def kick_off_each(self):
		db = self.db
		original_name = db['data_name']

		for i in range(1, 11):
			db['data_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '.csv'
			db['label_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '_label.csv'

			db['validation_data_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '_validation.csv'
			db['validation_label_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '_label_validation.csv'

			db['data_name'] = original_name + '_' + str(i)

			tmp_path = './tmp/' + db['data_name'] + '/'
			db_output_path = tmp_path + 'db_files/'
			batch_output_path = tmp_path + 'batch_outputs/'
			ensure_path_exists('./tmp')
			ensure_path_exists('./results')
			ensure_path_exists(tmp_path)
			ensure_path_exists(db_output_path)
			ensure_path_exists(batch_output_path)

			fname = self.output_db_to_text()
			self.export_bash_file(i, db['data_name'], fname)

			if socket.gethostname().find('login') != -1:
				call(["sbatch", "execute_combined.bash"])
				#call(["./src/hsic_algorithms.py", fname])
			else:
				os.system("bash ./execute_combined.bash")
		
	def collect_10_fold_info(self, W_optimize_technique):
		db = self.db
		original_name = db['data_name']

		nmi = []
		acc = []
		time = []
		cost = []

		for i in range(1,11):
			fin = open('./results/LSDR_' + original_name + '_' + str(i) + '_' + W_optimize_technique.__name__ + '.txt', 'r') 
			lns = fin.readlines()
			for line in lns:
				lst = line.split(':')
				if lst[0].strip() == 'NMI':
					nmi.append(float(lst[1]))
				if lst[0].strip() == 'ACC':
					acc.append(float(lst[1]))
				if lst[0].strip() == 'TIME':
					time.append(float(lst[1]))
				if lst[0].strip() == 'COST':
					cost.append(float(lst[1]))
			
		fin = open('./results/LSDR_' + original_name + '_batch_' + W_optimize_technique.__name__ + '.txt', 'w') 
		fin.write('NMI : \n\t%s\n'%str(nmi))
		fin.write('ACC : \n\t%s\n'%str(acc))
		fin.write('COST : \n\t%s\n'%str(cost))
		fin.write('TIME : \n\t%s\n\n'%str(time))

		fin.write('mean NMI : %.3f\n'%np.mean(np.array(nmi)))
		fin.write('mean ACC : %.3f\n'%np.mean(np.array(acc)))
		fin.write('mean COST : %.4f\n'%np.mean(np.array(cost)))
		fin.write('mean TIME : %.5f\n\n'%np.mean(np.array(time)))

		fin.write('std NMI : %.3f\n'%np.std(np.array(nmi)))
		fin.write('std ACC : %.3f\n'%np.std(np.array(acc)))
		fin.write('std COST : %.4f\n'%np.std(np.array(cost)))
		fin.write('std TIME : %.5f\n\n'%np.std(np.array(time)))

		fin.close()

		print(nmi)
		print(acc)
		print(cost)
		print(time)

		print('mean NMI : %.3f\n'%np.mean(np.array(nmi)))
		print('mean ACC : %.3f\n'%np.mean(np.array(acc)))
		print('mean COST : %.4f\n'%np.mean(np.array(cost)))
		print('mean TIME : %.5f\n\n'%np.mean(np.array(time)))

		print('std NMI : %.3f\n'%np.std(np.array(nmi)))
		print('std ACC : %.3f\n'%np.std(np.array(acc)))
		print('std COST : %.4f\n'%np.std(np.array(cost)))
		print('std TIME : %.5f\n\n'%np.std(np.array(time)))


	def gen_10_fold_data(self):
		db = self.db

		db['data_file_name'] = './datasets/' + db['data_name'] + '.csv'
		db['label_file_name'] = './datasets/' + db['data_name'] + '_label.csv'
		loader = db['dataset_class'](db)
		X = loader.X
		Y = loader.Y

		fold_path = './datasets/' + db['data_name'] + '/'
		ensure_path_exists(fold_path)
		count = 1

		kf = KFold(n_splits=10, shuffle=True)
		kf.get_n_splits(loader.X)
		for train_index, test_index in kf.split(X):
			#print(train_index)
			#print(test_index, '\n')
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '.csv', X_train, delimiter=',', fmt='%.4f') 
			np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '_label.csv', Y_train, delimiter=',', fmt='%d') 
			np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '_validation.csv', X_train, delimiter=',', fmt='%.4f') 
			np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '_label_validation.csv', Y_train, delimiter=',', fmt='%d') 

			count += 1

