import common

import tensorflow as tf

import keras

from keras.models import  load_model
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import data_load

tf.set_random_seed(42)

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_string('model_path', 'BM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are saved')
flags.DEFINE_boolean('proxy_data', False , 'If this is being used to generate training data for proxy model')
flags.DEFINE_string('proxy_x', 'PX.npy', 'Path where proxy training data is to be saved')
flags.DEFINE_string('proxy_y', 'PY.npy', 'Path where proxy training data labels are to be saved')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')


def main(argv=None):
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)()

	if dataObject is None:
		print "Invalid dataset; exiting"
		exit()

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

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
