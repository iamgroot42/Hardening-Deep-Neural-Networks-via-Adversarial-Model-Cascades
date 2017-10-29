import common
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras
import tensorflow as tf
from cleverhans.attacks import VirtualAdversarialMethod 
from cleverhans.utils_keras import KerasModelWrapper

import data_load

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('xi', 1e-6, 'xi')
flags.DEFINE_integer('num_iters', 1, 'Number of iterations')
flags.DEFINE_float('eps', 2.0, 'Epsilon')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('data_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('data_y', 'ADY.npy', 'Path where original labels are to be saved')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_string('mode', 'attack', 'Whethere attacking model or generating data for hardening')


def main(argv=None):
	# Initialize data object
	keras.layers.core.K.set_learning_phase(0)
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)()

	if dataObject is None:
		print "Invalid dataset; exiting"
		exit()

	if FLAGS.mode == 'attack':# Set seed for reproducability
		(X, Y) = dataObject.get_attack_data()
	else:
		(X, Y) = dataObject.get_hardening_data()

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	raw_model = keras.models.load_model(FLAGS.model_path)
	model = KerasModelWrapper(raw_model)

	virtual = VirtualAdversarialMethod(model, sess=keras.backend.get_session())
	adv_x = virtual.generate_np(X, xi=FLAGS.xi, num_iterations=FLAGS.num_iters, eps=FLAGS.eps)

	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = raw_model.evaluate(adv_x, Y, batch_size=FLAGS.batch_size)
	print('\nError on adversarial examples: ' + str(1.0 - accuracy[1]))

	np.save(FLAGS.data_x, adv_x)
	np.save(FLAGS.data_y, Y)


if __name__ == '__main__':
	app.run()
