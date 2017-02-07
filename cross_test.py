from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

import keras
import json
from keras.models import model_from_json
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import tf_model_eval
import utils_mnist, utils_cifar
import utils
from sklearn.externals import joblib
import vbow
import nn_svm

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('model_path', 'BM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')
flags.DEFINE_string('cluster', 'C.pkl', 'Path where cluster/SVM model is saved')
flags.DEFINE_string('arch', 'arch.json', 'Path where cluster/SVM model is to be saved')


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

	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	if FLAGS.is_autoencoder == 3:
		with open(FLAGS.arch) as data_file:
			model = model_from_json(json.load(data_file))
		cluster = joblib.load(FLAGS.cluster)
		model.load_weights(FLAGS.model_path)
		err = nn_svm.hybrid_error(X_test_adv, Y_test, model, cluster)
		print('Misclassification accuracy on adversarial examples: ' + str(1-err))
	else:
		if FLAGS.is_autoencoder == 2:
			cluster = joblib.load(FLAGS.cluster)
			x_shape, y_shape = utils_cifar.placeholder_shapes_handpicked(cluster.n_clusters)
			X_test_adv = X_test_adv.reshape(X_test_adv.shape[0], 32, 32, 3)
			X_test_adv = vbow.img_to_vect(X_test_adv, cluster)

		x = tf.placeholder(tf.float32, shape=x_shape)
		y = tf.placeholder(tf.float32, shape=y_shape)

		model = utils.load_model(FLAGS.model_path)
		predictions = model(x)

		accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
		print('Misclassification accuracy on adversarial examples: ' + str(1.0 - accuracy))


if __name__ == '__main__':
	app.run()

