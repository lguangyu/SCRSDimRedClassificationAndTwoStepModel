#!/usr/bin/env python3

import abc


class ABCClassifier(abc.ABC):
	@abc.abstractmethod
	def __init__(self, *ka, **kw):
		super(ABCClassifier, self).__init__(*ka, **kw)
		return

	@abc.abstractmethod
	def train(self, *ka, **kw):
		return super(ABCClassifier, self).train(*ka, **kw)

	@abc.abstractmethod
	def predict(self, *ka, **kw):
		return super(ABCClassifier, self).predict(*ka, **kw)


class ABCDimensionReducer(abc.ABC):
	@property
	def require_scale(self):
		return self._require_scale_

	@abc.abstractmethod
	def __init__(self, *ka, **kw):
		super(ABCDimensionReducer, self).__init__(*ka, **kw)
		return

	@abc.abstractmethod
	def train(self, *ka, **kw):
		return super(ABCDimensionReducer, self).train(*ka, **kw)

	@abc.abstractmethod
	def transform(self, *ka, **kw):
		return super(ABCDimensionReducer, self).transform(*ka, **kw)

	def train_transform(self, *ka, **kw):
		self.train(*ka, **kw)
		return self.transform(*ka, **kw)


class ABCModel(abc.ABC):
	@abc.abstractmethod
	def __init__(self, *ka, **kw):
		pass

	@abc.abstractmethod
	def train(self, *ka, **kw):
		pass

	@abc.abstractmethod
	def predict(self, *ka, **kw):
		pass

	@abc.abstractmethod
	def test(self, *ka, **kw):
		pass
