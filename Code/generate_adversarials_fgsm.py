import common

import keras
import numpy as np

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import batch_eval
import utils_mnist, utils_cifar
import helpers

from keras.models import load_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('fgsm_eps', 0.1, 'Tunable parameter for FGSM')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')

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

	x_shape, y_shape = utils_cifar.placeholder_shapes()

	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)
	
	print("Starting generation for epsilon = " + str(FLAGS.fgsm_eps))
	model = load_model(FLAGS.model_path)
	X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=n_classes)
	np.save(FLAGS.adversary_path_xo, X_test_pm)
	# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
	predictions = model(x)
	adv_x = helpers.fgsm(x, predictions, eps=FLAGS.fgsm_eps)
	X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test_pm])
	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = model.evaluate(X_test_adv, Y_test_pm, batch_size=FLAGS.batch_size)
	print('\nMisclassification accuracy on adversarial examples: ' + str((1.0 - accuracy[1])*100))
	np.save(FLAGS.adversary_path_x, X_test_adv)
	np.save(FLAGS.adversary_path_y, Y_test_pm)


if __name__ == '__main__':
	app.run()
