#!/usr/bin/env python3

import argparse


class CVShuffleParam(dict):
	def __init__(self, value):
		self.clear()
		if value == "random":
			self.update({"shuffle": True, "random_state": None})
		elif value == "disable":
			self.update({"shuffle": False})
		else:
			self.update({"shuffle": True, "random_state": int(value)})
		return
