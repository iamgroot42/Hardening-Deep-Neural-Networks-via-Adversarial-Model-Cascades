from abc import ABCMeta

import tensorflow as tf
import keras

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


class Model(object):
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	def __call__(self, *args, **kwargs):
		return self.get_probs(*args, **kwargs)

	def get_layer(self, x, layer):
		output = self.fprop(x)
		try:
			requested = output[layer]
		except KeyError:
			raise NoSuchLayerError()
		return requested

	def get_logits(self, x):
		return self.get_layer(x, 'logits')

	def get_probs(self, x):
		try:
			return self.get_layer(x, 'probs')
		except NoSuchLayerError:
			return tf.nn.softmax(self.get_logits(x))

	def get_layer_names(self):
		if hasattr(self, 'layer_names'):
			return self.layer_names

	def fprop(self, x):
		raise NotImplementedError('`fprop` not implemented.')


class KerasModelWrapper(Model):
	def __init__(self, model=None):
		super(KerasModelWrapper, self).__init__()
		self.model = model
		self.keras_model = None

	def _get_softmax_name(self):
		for i, layer in enumerate(self.model.layers):
			cfg = layer.get_config()
			if 'activation' in cfg and cfg['activation'] == 'softmax':
				return layer.name
		raise Exception("No softmax layers found")

	def _get_logits_name(self):
		softmax_name = self._get_softmax_name()
		softmax_layer = self.model.get_layer(softmax_name)
		node = softmax_layer.inbound_nodes[0]
		logits_name = node.inbound_layers[0].name
		return logits_name

	def get_logits(self, x):
		logits_name = self._get_logits_name()
		return self.get_layer(x, logits_name)

	def get_probs(self, x):
		name = self._get_softmax_name()
		return self.get_layer(x, name)

	def get_layer_names(self):
		layer_names = [x.name for x in self.model.layers]
		return layer_names

	def fprop(self, x):
		from keras.models import Model as KerasModel

		if self.keras_model is None:
			new_input = self.model.get_input_at(0)
			out_layers = [x_layer.output for x_layer in self.model.layers]
			self.keras_model = KerasModel(new_input, out_layers)
		outputs = self.keras_model(x)
		if len(self.model.layers) == 1:
			outputs = [outputs]
		fprop_dict = dict(zip(self.get_layer_names(), outputs))
		return fprop_dict

