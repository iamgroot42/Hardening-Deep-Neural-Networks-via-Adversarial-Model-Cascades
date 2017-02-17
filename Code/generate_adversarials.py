from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

import keras
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import tf_model_eval, batch_eval
import utils_mnist, utils_cifar
import helpers
import utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('fgsm_eps', 0.1, 'Tunable parameter for FGSM')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')
flags.DEFINE_integer('per_class_adv', 200 , 'Number of adversarial examples to be picked per class')


def main(argv=None):
	flatten = False
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	# config = tf.ConfigProto(
	#	device_count = {'GPU': 0}
	# )
	# sess = tf.Session(config=config)
	sess = tf.Session()
	keras.backend.set_session(sess)
	# Get MNIST test data
	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()

	if flatten:
		X_test = X_test.reshape(10000, 784)

	label_smooth = .1
	Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

	if flatten:
		x_shape, y_shape = utils_mnist.placeholder_shapes_flat()
	else:
		x_shape, y_shape = utils_cifar.placeholder_shapes()

	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)

	model = utils.load_model(FLAGS.model_path)
	predictions = model(x)
	X_test, Y_test = helpers.jbda(X_test, Y_test, FLAGS.per_class_adv)
	# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
	adv_x = helpers.fgsm(x, predictions, eps=FLAGS.fgsm_eps)
	X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
	print('Misclassification accuracy on adversarial examples: ' + str(1.0 - accuracy))
	np.save(FLAGS.adversary_path_x, X_test_adv)
	np.save(FLAGS.adversary_path_xo, X_test)
	np.save(FLAGS.adversary_path_y, Y_test)


if __name__ == '__main__':
	app.run()
