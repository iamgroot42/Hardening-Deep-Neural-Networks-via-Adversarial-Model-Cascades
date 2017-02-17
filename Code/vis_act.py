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

from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', 'BM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')
flags.DEFINE_string('cluster', 'C.pkl', 'Path where cluster/SVM model is saved')
flags.DEFINE_string('arch', 'arch.json', 'Path where cluster/SVM model is to be saved')


def visualize_path(model, point):
	for layer in model.layers[1:-1]:
		if layer.name.split('_')[0] in ["dropout"]:
			continue
		partial = Model(input=model.inputs, output=layer.output)
		partial.compile(loss='binary_crossentropy',optimizer='Adadelta')
		output = partial.predict(point)[0]
		if len(output.shape) is 1:
			output = np.reshape(output, (output.shape[0]/32, 32))
			plt.matshow(output)
			plt.show()
		else:
			# for filter_index in range(output.shape[0]):
			for filter_index in range(1):
				show_this = output[filter_index,:,:]
				plt.matshow(show_this)
				plt.show()
	return output
		


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

	X_test = np.load(FLAGS.adversary_path_xo)
	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	if FLAGS.is_autoencoder == 3:
		with open(FLAGS.arch) as data_file:
			model = model_from_json(json.load(data_file))
		cluster = joblib.load(FLAGS.cluster)
		model.load_weights(FLAGS.model_path)

		counter = 1 #1, 25, 
		orig_point = X_test.astype('float32')[counter:counter+1]
		point = X_test_adv.astype('float32')[counter:counter+1]
		while True:
			orig_point = X_test.astype('float32')[counter:counter+1]
			point = X_test_adv.astype('float32')[counter:counter+1]
			if cluster.predict(model.predict(orig_point))[0] != cluster.predict(model.predict(point)):
				break
			# if cluster.predict(model.predict(orig_point))[0] == cluster.predict(model.predict(point)):
			# 	if cluster.predict(model.predict(orig_point))[0] == np.argmax(Y_test[counter]):
			# 		break
			counter += 1
		
		print("Path taken for original sample")
		orig = visualize_path(model, orig_point)
		print("Path taken for moisy sample")
		noisy = visualize_path(model, point)

		print("Model's output:",cluster.predict(model.predict(orig_point))[0])
		print("Adversarial output:",cluster.predict(model.predict(point))[0])
		print("Actual label:", np.argmax(Y_test[counter]))

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

		counter = 0
		while True:
			orig_point = X_test.astype('float32')[counter:counter+1]
			point = X_test_adv.astype('float32')[counter:counter+1]
			if np.argmax(model.predict(orig_point)) != np.argmax(model.predict(point)):
				break
			counter += 1

		print("Path taken for original sample")
		visualize_path(model, orig_point)
		print("Path taken for moisy sample")
		visualize_path(model, point)

		print("Model's output:",np.argmax(model.predict(orig_point)))
		print("Adversarial output:",np.argmax(model.predict(point)))
		print("Actual label:", np.argmax(Y_test[counter]))


if __name__ == '__main__':
	app.run()
