import common

import keras
import numpy as np

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import batch_eval
import utils_mnist, utils_cifar, utils_svhn
import helpers

from keras.models import load_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('fgsm_eps', 0.1, 'Tunable parameter for FGSM')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where original labels are to be saved')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')


def main(argv=None):
	n_classes = 10
	if FLAGS.dataset == 'cifar100':
		n_classes = 100
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
		x_shape, y_shape = utils_cifar.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=n_classes)
	elif FLAGS.dataset == 'mnist':
		X_train, Y_train, X_test, Y_test = utils_mnist.data_mnist()
		x_shape, y_shape = utils_mnist.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=800, nb_classes=n_classes)
	elif FLAGS.dataset == 'svhn':
		X_train, Y_train, X_test, Y_test = utils_svhn.data_svhn()
		x_shape, y_shape = utils_svhn.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=1500, nb_classes=n_classes)
	else:
		print "Invalid dataset. Exiting."
		exit()

	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)

	print("Starting generation for epsilon = " + str(FLAGS.fgsm_eps))
	model = load_model(FLAGS.model_path)

	np.save(FLAGS.adversary_path_xo, X_test_pm)
	# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
	predictions = model(x)

	if FLAGS.dataset == 'cifar100':
		adv_x = helpers.fgsm(x, predictions, eps=FLAGS.fgsm_eps, clip_min=0.0, clip_max=1.0)
	else:
		adv_x = helpers.fgsm(x, predictions, eps=FLAGS.fgsm_eps, clip_min=0, clip_max=255)

	X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test_pm])
	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = model.evaluate(X_test_adv, Y_test_pm, batch_size=FLAGS.batch_size)
	print('\nMisclassification accuracy on adversarial examples: ' + str((1.0 - accuracy[1])*100))
	np.save(FLAGS.adversary_path_x, X_test_adv)
	np.save(FLAGS.adversary_path_y, Y_test_pm)


if __name__ == '__main__':
	app.run()
