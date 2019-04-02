#!/usr/bin/env python3

import os
import sys


class Logger(object):
	@property
	def fh(self):
		return self._fh

	def __init__(self, *ka, **kw):
		super(Logger, self).__init__()
		self.open(*ka, **kw)
		return

	def open(self, fname, mode = "w", **kw):
		"""
		open file handler, close previous one if exists
		"""
		try:
			self.close()
		except:
			pass
		self._fh = open(fname, mode, **kw)
		return

	def close(self):
		self.fh.close()
		return

	def __enter__(self):
		return self

	def __exit__(self, except_t, except_v, traceback):
		self.close()
		return # return None for not handled exception

	def print(self, msg, **kw):
		"""
		short cut to call print(msg, file = self.fh, **kw)
		"""
		print(msg, file = self.fh, **kw)
		return

	def tee(self, msg, **kw):
		"""
		short cut to call self.print, in addition to print identically to stdout
		"""
		self.print(msg, **kw)
		print(msg, file = sys.stdout, **kw)
		return
