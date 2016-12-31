from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

import keras
import sys

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import tf_model_train, tf_model_eval
import utils_mnist
import utils
import helpers

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('save_here', 'saved_model', 'Path where model is to be saved')


def main(argv=None):
	flatten = False
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)
	# Get MNIST test data
	
	X_train, Y_train, X_test, Y_test = utils_mnist.data_mnist()

	if flatten:
		X_train = X_train.reshape(60000, 784)
		X_test = X_test.reshape(10000, 784)

	label_smooth = .1
	Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

	if flatten:
		x_shape, y_shape = utils_mnist.placeholder_shapes_flat()
	else:
		x_shape, y_shape = utils_mnist.placeholder_shapes()

	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)
	model = utils_mnist.modelB()
	predictions = model(x)
	# model = utils_mnist.modelA()
	# predictions = model(x)

	X_train_p, Y_train_p = helpers.jbda(X_train, Y_train)
	# X_train_p, Y_train_p = X_train, Y_train
	# Train blackbox model
	tf_model_train(sess, x, y, predictions, X_train_p, Y_train_p)
	accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
	print('Test accuracy for blackbox model: ' + str(accuracy))
	utils.save_model(model, FLAGS.save_here)	


if __name__ == '__main__':
	app.run()
