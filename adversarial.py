from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

import keras

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_mnist import data_mnist, model_mnist
from utils_tf import tf_model_train, tf_model_eval, batch_eval
import fgsm

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
	# Set TF random seed to improve reproducibility
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)
	# Get MNIST test data
	X_train, Y_train, X_test, Y_test = data_mnist()
	assert Y_train.shape[1] == 10.
	label_smooth = .1
	Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

	# Define input TF placeholder (for blackbox model)
	x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
	y = tf.placeholder(tf.float32, shape=(None, 10))

	# Define TF model graph (for blackbox model)
	model = model_mnist()
	predictions = model(x)

	def evaluate():
		# Evaluate the accuracy of the MNIST model on legitimate test examples
		accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
		print('Test accuracy on legitimate test examples: ' + str(accuracy))

	# print("Jacobian: ")
	# J = tf.test.compute_gradient(x, (None, 1, 28, 28), Y_test, Y_test.shape)
	# print(J)	

	# ToDo: Replace training data for proxy with Jacobian Augmented Data
	X_train_p = X_train[:100,:,:,:]
	Y_train_p = Y_train[:100:]
	X_train = X_train[:500,:,:,:]
	Y_train = Y_train[:500:]
	X_test = X_test[:200]
	Y_test = Y_test[:200]

	# Train an MNIST model (blackbox model)
	tf_model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate)

	# Define TF model graph (for proxy model)
	model_p = model_mnist()
	predictions_p = model(x)

	# Train an MNIST model (proxy model)
	tf_model_train(sess, x, y, predictions_p, X_train_p, Y_train_p, evaluate=evaluate)

	# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
	adv_x = fgsm.fgsm(x, predictions, eps=0.3)
	X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
	# Evaluate the accuracy of the MNIST model on adversarial examples
	accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
	print('Misclassification accuracy on adversarial examples (black box): ' + str(1.0 - accuracy))

	# Craft adversarial examples from proxy network and check their accuracy on black box
	adv_x_p = fgsm.fgsm(x, predictions_p, eps=0.3)
	X_test_adv_p, = batch_eval(sess, [x], [adv_x_p], [X_test])
	# Evaluate the accuracy of the MNIST model on adversarial examples
	accuracy_p = tf_model_eval(sess, x, y, predictions_p, X_test_adv_p, Y_test)
	print('Misclassification accuracy on adversarial examples (proxy): ' + str(1.0 - accuracy_p))

	# Check classification accuracy of adversarial examples of proxy on black box
	accuracy_bp = tf_model_eval(sess, x, y, predictions, X_test_adv_p, Y_test)
	print('Misclassification accuracy_bp on adversarial examples on blackbox from proxy: ' + str(1.0 - accuracy_bp))


if __name__ == '__main__':
	app.run()
