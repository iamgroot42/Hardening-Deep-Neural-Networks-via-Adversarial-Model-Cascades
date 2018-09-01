import common
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras
import tensorflow as tf

import data_load

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_string('model_path', '', 'Path where model is stored')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_string('test_prefix', "", 'Prefix path for custom generated test data')


def main(argv=None):
	# Initialize data object
	keras.layers.core.K.set_learning_phase(0)
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)()

	if dataObject is None:
		print("Invalid dataset; exiting")
		exit()

	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
	model = keras.models.load_model(FLAGS.model_path)
	if len(FLAGS.test_prefix):
		X_test, Y_test = np.load(FLAGS.test_prefix + "_x.npy"), np.load(FLAGS.test_prefix + "_y.npy")
		print("Custom test data found")

	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = model.evaluate(X_test, Y_test, batch_size=FLAGS.batch_size)
	print('\nTest accuracy: ' + str(accuracy[1]))

if __name__ == '__main__':
	app.run()
