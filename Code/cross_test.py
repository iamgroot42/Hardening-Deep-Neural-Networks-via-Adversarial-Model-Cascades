import common

import tensorflow as tf

import keras
import json
from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import utils_mnist, utils_cifar
import utils
from sklearn.externals import joblib
from Models import vbow, nn_svm
import helpers

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('model_path', 'BM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')
flags.DEFINE_string('cluster', 'C.pkl', 'Path where cluster/SVM model is saved')
flags.DEFINE_string('arch', 'arch.json', 'Path where cluster/SVM model is to be saved')
flags.DEFINE_boolean('proxy_data', False , 'If this is being used to generate training data for proxy model')
flags.DEFINE_string('proxy_x', 'PX.npy', 'Path where proxy training data is to be saved')
flags.DEFINE_string('proxy_y', 'PY.npy', 'Path where proxy training data labels are to be saved')
flags.DEFINE_integer('per_class_adv', 10 , 'Number of adversarial examples to be picked per class')


def main(argv=None):
	nb_classes = 100
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)

	if FLAGS.is_autoencoder == 2 and FLAGS.is_blackbox:
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar_raw()
	else:
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()

	X_test_adv = None
	if not FLAGS.proxy_data:
		X_test_adv = np.load(FLAGS.adversary_path_x)
		Y_test = np.load(FLAGS.adversary_path_y)

	if FLAGS.is_autoencoder == 3:
		with open(FLAGS.arch) as data_file:
			model = model_from_json(json.load(data_file))
		cluster = joblib.load(FLAGS.cluster)
		model.load_weights(FLAGS.model_path)
		if FLAGS.proxy_data:
			X_train_p, Y_train_p = helpers.jbda(X_train, Y_train, "train", FLAGS.per_class_adv, nb_classes)
			Y_train_p = np_utils.to_categorical(nn_svm.get_output(X_train_p, model, cluster),nb_classes)
			np.save(FLAGS.proxy_x, X_train_p)
			np.save(FLAGS.proxy_y, Y_train_p)
			print('Proxy dataset created')
		else:
			err = nn_svm.hybrid_error(X_test_adv, Y_test, model, cluster)
			print('\nMisclassification accuracy on adversarial examples: ' + str(1-err))
	else:
		if FLAGS.is_autoencoder == 2:
			cluster = joblib.load(FLAGS.cluster)
			x_shape, y_shape = utils_cifar.placeholder_shapes_handpicked(cluster.n_clusters)
			if FLAGS.proxy_data:
				X_train_p, Y_train_p = helpers.jbda(X_train, Y_train, "train", FLAGS.per_class_adv, nb_classes)
				X_test_adv = X_train_p
			X_test_adv = X_test_adv.reshape(X_test_adv.shape[0], 32, 32, 3)
			X_test_adv = vbow.img_to_vect(X_test_adv, cluster)
		elif FLAGS.proxy_data:
			X_train_p, Y_train_p = helpers.jbda(X_train, Y_train, "train", FLAGS.per_class_adv, nb_classes)
			X_test_adv = X_train_p

		model = utils.load_model(FLAGS.model_path)

		if FLAGS.proxy_data:
			Y_train_p = model.predict(X_test_adv)
			np.save(FLAGS.proxy_x, X_train_p)
			np.save(FLAGS.proxy_y, Y_train_p)
			print('Proxy dataset created')
		else:
			accuracy = model.evaluate(X_test_adv, Y_test, batch_size=FLAGS.batch_size)
			print('\nMisclassification accuracy on adversarial examples: ' + str(100*(1.0 - accuracy[1])))


if __name__ == '__main__':
	app.run()
