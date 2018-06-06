import common
import keras

from cleverhans.utils_tf import model_train, model_eval
from tensorflow.python.platform import app
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

import data_load
import helpers
from Models import resnet, sota

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_integer('nb_epochs', 300, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-1, 'Learning rate for training')
flags.DEFINE_string('save_here', 'dummy.h5', 'Path where model is to be saved')
flags.DEFINE_string('attack_name', 'fgsm', 'Name of attack against which adversarial hardening is to be performed')
flags.DEFINE_integer('stack_n', 5, 'stack number n, total layers = 6 * n + 2 (default: 5)')

# Epoch number (global for function referece)
epoch_number = 0

def main(argv=None):
	# Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'th':
                keras.backend.set_image_dim_ordering('th')

	# Object used to keep track of (and return) key accuracies
	report = AccuracyReport()

	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)

	# Black-box network
	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
	X_train, Y_train, X_validation, Y_validation = dataObject.validation_split(blackbox_Xtrain, blackbox_Ytrain, 0.2)
	n_classes = Y_train.shape[1]

	# Set learning phase to testing
	keras.backend.set_learning_phase(False)

	# Define number of iterations
	iterations         = 50000 // FLAGS.batch_size + 1

	model, _ = resnet.residual_network(n_classes=n_classes, stack_n=FLAGS.stack_n)
	#model = sota.get_appropriate_model(FLAGS.dataset)(FLAGS.learning_rate, n_classes)
	wrap = KerasModelWrapper(model)

	train_params = {
        	'nb_epochs': FLAGS.nb_epochs,
	        'batch_size': FLAGS.batch_size,
        	'learning_rate': FLAGS.learning_rate
    	}

	# x,y : get appropriate placeholders in x and y
	shapes = dataObject.get_placeholder_shape()
	x, y = tf.placeholder(tf.float32, shape=shapes[0]), tf.placeholder(tf.float32, shape=shapes[1])
	preds = model(x)

	def evaluate():
		global epoch_number
	        # Evaluate the accuracy of the MNIST model on legitimate test examples
        	eval_params = {'batch_size': FLAGS.batch_size}
	        acc = model_eval(common.sess, x, y, preds, X_validation, Y_validation, args=eval_params)
        	report.clean_train_clean_eval = acc
        	print('Validation accuracy %0.4f' % acc)
		# Change learning rate according to scheduler
		new_lr = resnet.scheduler(epoch_number)
		epoch_number += 1
		model.optimizer.lr.assign(new_lr)
		print('Learning rate after this epoch %f' % K.eval(model.optimizer.lr))

	# Define attack and its parameters
	attack, attack_params = helpers.get_approproiate_attack(FLAGS.dataset, FLAGS.attack_name
		,wrap, common.sess, harden=True, attack_type="None")

	# Run adversarial training
	adversarial_predictions = model(attack.generate(x, **attack_params))
	model_train(common.sess, x, y, preds, X_train, Y_train
		,save=False, predictions_adv=adversarial_predictions
		,args=train_params, evaluate=evaluate)

	# Save model
	model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
