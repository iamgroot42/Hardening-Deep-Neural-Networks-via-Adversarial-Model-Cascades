import copy
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras
import attacks_tf ,attacks
import helpers
from utils_tf import batch_eval
import utils_cifar, utils_mnist, utils_svhn

from keras_to_ch import KerasModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_subset_classes', 10, 'Number of target classes')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon')


def main(argv=None):
	n_classes = 10
	image_shape = (32, 32, 3)
	if FLAGS.dataset == 'cifar100':
		n_classes = 100
		_, _, X_test, Y_test = utils_cifar.data_cifar()
		x_shape, y_shape = utils_cifar.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=10, nb_classes=n_classes)
	elif FLAGS.dataset == 'mnist':
		_, _, X_test, Y_test = utils_mnist.data_mnist()
		x_shape, y_shape = utils_mnist.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=n_classes)
		image_shape = (28, 28, 1)
	elif FLAGS.dataset == 'svhn':
		_, _, X_test, Y_test = utils_svhn.data_svhn()
		x_shape, y_shape = utils_svhn.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=n_classes)
	else:
		print("Invalid dataset. Exiting.")
		exit()

	keras.layers.core.K.set_learning_phase(0)

	tf.set_random_seed(1234)
	if keras.backend.image_dim_ordering() != 'tf':
		keras.backend.set_image_dim_ordering('tf')

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	raw_model = keras.models.load_model(FLAGS.model_path)
	model = KerasModelWrapper(raw_model)

	x = tf.placeholder(tf.float32, shape=x_shape)

	madry = attacks.MadryEtAl(model, sess=sess)
	adv_x = madry.generate_np(X_test_pm, eps=FLAGS.epsilon, clip_min=0.0, clip_max=1.0)

	accuracy = raw_model.evaluate(adv_x, Y_test_pm, batch_size=128)
        print('\nMisclassification accuracy on adversarial examples: ' + str((1.0 - accuracy[1])*100))

	np.save(FLAGS.adversary_path_y, Y_test_pm)
	np.save(FLAGS.adversary_path_x, adv_x)


if __name__ == '__main__':
	app.run()

