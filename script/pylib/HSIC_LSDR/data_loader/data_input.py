#!/usr/bin/env python

from ..test_base import *
import sklearn.metrics
import numpy as np
#from .DLoader import *
from .Pdataset import *
# fix missing DLoader, seems changed to Pdataset
DLoader = Pdataset

class data_input(DLoader):
	def __init__(self, db):
		self.dtype = np.float64				#np.float32
		self.X = db['X']
		if db['center_and_scale']: self.X = preprocessing.scale(self.X)

		if 'Y' in db:
			self.Y = db['Y']
		else:
			self.Y = None

		self.initilize_data_info(db)


