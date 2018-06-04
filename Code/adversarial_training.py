import common
import keras

from cleverhans.utils_tf import model_train, model_eval
from tensorflow.python.platform import app
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

import data_load
import helpers
from Models import sota

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_integer('nb_epochs', 300, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_string('save_here', 'dummy.h5', 'Path where model is to be saved')
flags.DEFINE_string('attack_name', 'fgsm', 'Name of attack against which adversarial hardening is to be performed')


def main(argv=None):
	# Object used to keep track of (and return) key accuracies
	report = AccuracyReport()

	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)

	# Black-box network
	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
	X_train, Y_train, X_validation, Y_validation = dataObject.validation_split(blackbox_Xtrain, blackbox_Ytrain, 0.2)
	n_classes = Y_train.shape[1]

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	# Set learning phase to testing
	keras.backend.set_learning_phase(False)

	raw_model = sota.get_appropriate_model(FLAGS.dataset)(FLAGS.learning_rate, n_classes)
	model = KerasModelWrapper(raw_model)

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
	        # Evaluate the accuracy of the MNIST model on legitimate test examples
        	eval_params = {'batch_size': FLAGS.batch_size}
	        acc = model_eval(common.sess, x, y, preds, X_validation, Y_validation, args=eval_params)
        	report.clean_train_clean_eval = acc
        	print('Validation accuracy %0.4f' % acc)

	# Define attack and its parameters
	attack, attack_params = helpers.get_approproiate_attack(FLAGS.dataset, FLAGS.attack_name
		,model, common.sess, harden=True, attack_type="None")

	# Run adversarial training
	adversarial_predictions = model(attack.generate(x, **attack_params))
	model_train(common.sess, x, y, preds, X_train, Y_train
		,save=False, predictions_adv=adversarial_predictions
		,args=train_params, evaluate=evaluate)

	# Save model
	raw_model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
