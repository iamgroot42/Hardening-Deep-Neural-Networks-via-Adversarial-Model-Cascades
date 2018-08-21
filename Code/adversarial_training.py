import common
import keras

from utils_tf import tf_model_train, model_eval
from tensorflow.python.platform import app
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.python.platform import flags
from keras.models import Model

from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

import data_load
import helpers
from Models import resnet, sota, densenet

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-1, 'Learning rate for training')
flags.DEFINE_string('save_here', 'saved_model.h5', 'Path where model is to be saved')
flags.DEFINE_string('attack_name', 'fgsm', 'Name of attack against which adversarial hardening is to be performed')
flags.DEFINE_integer('stack_n', 5, 'stack number n, total layers = 6 * n + 2 (default: 5)')
flags.DEFINE_float('eta', 1, 'Adversarial regularization')

# Epoch number (global for function referece)
epoch_number = 0

def main(argv=None):
	# Set learning phase
	#keras.layers.core.K.set_learning_phase(0)

	# Object used to keep track of (and return) key accuracies
	report = AccuracyReport()

	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)
	datagen = dataObject.data_generator()

	# Black-box network
	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
	X_train, Y_train, X_validation, Y_validation = dataObject.validation_split(blackbox_Xtrain, blackbox_Ytrain, 0.2)
	datagen.fit(X_train)
	n_classes = Y_train.shape[1]

	#model, _ = resnet.residual_network(n_classes=n_classes, stack_n=FLAGS.stack_n)
	(x, y), _, _ = densenet.densenet(n_classes=n_classes, mnist=(FLAGS.dataset=="mnist"), get_logits=False)
	model = Model(x, y)
	#model = sota.get_appropriate_model(FLAGS.dataset)(FLAGS.learning_rate, n_classes)
	wrap = KerasModelWrapper(model)

	# x,y : get appropriate placeholders in x and y
	shapes = dataObject.get_placeholder_shape()
	#x, y = tf.placeholder(tf.float32, shape=shapes[0]), tf.placeholder(tf.float32, shape=shapes[1])
	#x, y = model.input, model.output
	preds = model(x)

	def evaluate():
		global epoch_number
	        # Evaluate the accuracy of the MNIST model on legitimate test examples
        	eval_params = {'batch_size': FLAGS.batch_size}
	        acc = model_eval(common.sess, x, y, preds, FLAGS, X_validation, Y_validation)
        	report.clean_train_clean_eval = acc
		return acc
		#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		#	print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

	# Define attack and its parameters
	attack, attack_params = helpers.get_approproiate_attack(FLAGS.dataset, FLAGS.attack_name
		,wrap, common.sess, harden=True, attack_type="None")

	# Define optimizer to be used while running adversarial training
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

	# Run adversarial training
	adversarial_predictions = model(attack.generate(x, **attack_params))
	tf_model_train(common.sess, model, x, y, preds, X_train, Y_train, FLAGS
		,evaluate=evaluate, scheduler=densenet.scheduler,
		predictions_adv=adversarial_predictions, data_generator=datagen)
		#,optimizer=optimizer)

	# Save model
	model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
