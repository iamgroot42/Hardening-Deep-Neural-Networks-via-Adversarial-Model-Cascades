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

from utils_tf import tf_model_eval
import utils_mnist, utils_cifar
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('model_path', 'saved_model', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'adversaries_x.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'adversaries_y.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2) or none(0)')


def main(argv=None):
	flatten = False
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)

	if flatten:
		x_shape, y_shape = utils_mnist.placeholder_shapes_flat()
	else:
		x_shape, y_shape = utils_cifar.placeholder_shapes()

	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)

	model = utils.load_model(FLAGS.model_path)
	predictions = model(x)

	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	# Check classification accuracy of adversarial examples of proxy on black box
	accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
	print('Misclassification accuracy on adversarial examples: ' + str(1.0 - accuracy))


if __name__ == '__main__':
	app.run()
