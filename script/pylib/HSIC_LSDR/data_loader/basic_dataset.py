#!/usr/bin/env python

from ..test_base import *
import sklearn.metrics
import numpy as np
from .Pdataset import *

class basic_dataset(Pdataset):
	def __init__(self, db):
		Pdataset.__init__(self, db)
		self.initilize_data_info(db)
