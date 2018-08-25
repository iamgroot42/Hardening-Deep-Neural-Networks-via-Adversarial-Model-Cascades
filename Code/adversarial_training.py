import common
import keras
import math
import sys

from utils_tf import tf_model_train, model_eval
from tensorflow.python.platform import app
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
import numpy as np
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


def main(argv=None):
	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)
	datagen = dataObject.data_generator()

	# Black-box network
	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
	X_train, Y_train, X_validation, Y_validation = dataObject.validation_split(blackbox_Xtrain, blackbox_Ytrain, 0.2)
	datagen.fit(X_train)
	n_classes = Y_train.shape[1]

	#model, _ = resnet.residual_network(n_classes=n_classes, stack_n=FLAGS.stack_n)
	_, model, _ = densenet.densenet(n_classes=n_classes, mnist=(FLAGS.dataset=="mnist"), get_logits=False)

	# Define attack and its parameters
	attack, attack_params = helpers.get_appropriate_attack(FLAGS.dataset, dataObject.get_range(), FLAGS.attack_name
		,KerasModelWrapper(model), common.sess, harden=True, attack_type="None")

	# Run adversarial training
	helpers.customTrainModel(model, X_train, Y_train, X_validation, Y_validation, datagen,
		FLAGS.nb_epochs, densenet.scheduler, FLAGS.batch_size,
		attacks=[(attack, attack_params)])

	# Save model
	model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
