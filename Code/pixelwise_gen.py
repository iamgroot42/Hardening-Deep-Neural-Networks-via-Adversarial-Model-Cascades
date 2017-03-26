import common

import keras
import numpy as np

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import utils_mnist, utils_cifar
import helpers
import perturb

from keras.models import load_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')
flags.DEFINE_integer('per_class_adv', 1000 , 'Number of adversarial examples to be picked per class')
flags.DEFINE_integer('p', 1 , 'p')
flags.DEFINE_integer('r', 1 , 'r')
flags.DEFINE_integer('d', 3 , 'd')
flags.DEFINE_integer('t', 10 , 't')
flags.DEFINE_integer('k', 2 , 'k')
flags.DEFINE_integer('R', 4 , 'R')


def main(argv=None):
	n_classes = 100
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)

	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()

	label_smooth = .1
	Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

	model = load_model(FLAGS.model_path)
	X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=FLAGS.per_class_adv, nb_classes=n_classes)
	
	X_test_adv, Y_test_adv = perturb.perturb_images(model, X_test_pm, Y_test_pm, FLAGS.p, FLAGS.r, FLAGS.d, FLAGS.t, FLAGS.k, FLAGS.R)

	# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
	np.save(FLAGS.adversary_path_x, X_test_adv)
	np.save(FLAGS.adversary_path_xo, X_test_pm)
	np.save(FLAGS.adversary_path_y, Y_test_adv)


if __name__ == '__main__':
	app.run()
