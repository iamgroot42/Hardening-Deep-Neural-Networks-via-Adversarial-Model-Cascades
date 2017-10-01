import common

import tensorflow as tf

import keras
import json
from keras.models import model_from_json, load_model, Model
from keras.layers import Flatten
from keras.utils import np_utils
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import utils_mnist, utils_cifar, utils_svhn
import utils
from sklearn.externals import joblib
import helpers

FLAGS = flags.FLAGS


flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_string('model_path', 'BM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are saved')
flags.DEFINE_boolean('proxy_data', False , 'If this is being used to generate training data for proxy model')
flags.DEFINE_string('proxy_x', 'PX.npy', 'Path where proxy training data is to be saved')
flags.DEFINE_string('proxy_y', 'PY.npy', 'Path where proxy training data labels are to be saved')
flags.DEFINE_integer('per_class_adv', 10 , 'Number of adversarial examples to be picked per class')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')


def main(argv=None):
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	if FLAGS.dataset == 'cifar100':
		nb_classes = 100
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
		X_train_bm, Y_train_bm, X_train_pm, Y_train_pm = helpers.jbda(X_train, Y_train, "train", 500, nb_classes)
	elif FLAGS.dataset == 'mnist':
		nb_classes = 10
		X_train, Y_train, X_test, Y_test = utils_mnist.data_mnist()
		X_train_pm, Y_train_pm, X_test_pm, Y_test_pm = helpers.jbda(X_train, Y_train, "train", 5000, nb_classes)
	elif FLAGS.dataset == 'svhn':
		nb_classes = 10
		X_train, Y_train, X_test, Y_test = utils_svhn.data_svhn()
		X_train_pm, Y_train_pm, X_test_pm, Y_test_pm = helpers.jbda(X_train, Y_train, "train", 300, nb_classes)

	X_train_pm, Y_train_pm, X_test_pm, Y_test_pm = helpers.jbda(X_train_pm, Y_train_pm, "train", FLAGS.per_class_adv, nb_classes)

	X_test_adv = None
	if not FLAGS.proxy_data:
		X_test_adv = np.load(FLAGS.adversary_path_x)
		Y_test = np.load(FLAGS.adversary_path_y)

	model = utils.load_model(FLAGS.model_path)
	if FLAGS.proxy_data:
		Y_train_p = model.predict(X_train_pm)
		np.save(FLAGS.proxy_x, X_train_pm)
		np.save(FLAGS.proxy_y, Y_train_p)
		print('Proxy dataset created')
	else:
		accuracy = model.evaluate(X_test_adv, Y_test, batch_size=FLAGS.batch_size)
		print('\nMisclassification accuracy on adversarial examples: ' + str(100*(1.0 - accuracy[1])))


if __name__ == '__main__':
	app.run()
