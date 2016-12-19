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

from utils_tf import tf_model_train, tf_model_eval, batch_eval
import utils_mnist
import advhelp

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)
	# Get MNIST test data
	X_train, Y_train, X_test, Y_test = utils_mnist.data_mnist()
	label_smooth = .1
	Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

	x_shape, y_shape = utils_mnist.placeholder_shapes()
	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)
	model = utils_mnist.blackbox_model()
	predictions = model(x)
	model_p = utils_mnist.proxy_model()
	predictions_p = model_p(x)
	
	# J = tf.test.compute_gradient(x, (None, 1, 28, 28), Y_test, Y_test.shape)
	
	X_train_p, Y_train_p = advhelp.jbda(X_train, Y_train)
	X_train = X_train[:500,:,:,:]
	Y_train = Y_train[:500]
	X_test = X_test[:500,:,:,:]
	Y_test = Y_test[:500]
	# Train blackbox model
	tf_model_train(sess, x, y, predictions, X_train, Y_train)
	accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
	print('Test accuracy for blackbox model: ' + str(accuracy))
	# Train proxy model
	tf_model_train(sess, x, y, predictions_p, X_train, Y_train)
	accuracy = tf_model_eval(sess, x, y, predictions_p, X_test, Y_test)
	print('Test accuracy for proxy model: ' + str(accuracy))

	# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
	adv_x = advhelp.fgsm(x, predictions, eps=0.3)
	X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
	print('Misclassification accuracy on adversarial examples (black box): ' + str(1.0 - accuracy))

	# Craft adversarial examples from proxy network and check their accuracy on black box
	adv_x_p = advhelp.fgsm(x, predictions_p, eps=0.3)
	X_test_adv_p, = batch_eval(sess, [x], [adv_x_p], [X_test])
	# Evaluate the accuracy of the proxy model on adversarial examples
	accuracy_p = tf_model_eval(sess, x, y, predictions_p, X_test_adv_p, Y_test)
	print('Misclassification accuracy on adversarial examples (proxy): ' + str(1.0 - accuracy_p))

	# Check classification accuracy of adversarial examples of proxy on black box
	accuracy_bp = tf_model_eval(sess, x, y, predictions, X_test_adv_p, Y_test)
	print('Misclassification accuracy_bp on adversarial examples on blackbox from proxy: ' + str(1.0 - accuracy_bp))


if __name__ == '__main__':
	app.run()
