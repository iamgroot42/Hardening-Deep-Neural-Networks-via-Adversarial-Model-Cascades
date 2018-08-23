import common
import keras

from tensorflow.python.platform import app
from keras.models import load_model

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from cleverhans.utils_keras import KerasModelWrapper

import data_load
import helpers

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_string('model', 'saved_model.h5', 'Path where model is saved')
flags.DEFINE_string('attack_name', 'fgsm', 'Name of attack against which adversarial hardening is to be performed')
flags.DEFINE_string('save_here', None, 'Path to save perturbed examples')

def main(argv=None):
	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)
	datagen = dataObject.data_generator()

	# Load attack data
	attack_X, attack_Y = dataObject.get_attack_data()
	n_classes = attack_Y.shape[1]

	# Load model
	model = load_model(FLAGS.model)

	# Define attack and its parameters
	attack, attack_params = helpers.get_appropriate_attack(FLAGS.dataset, dataObject.get_range(), FLAGS.attack_name
		,KerasModelWrapper(model), common.sess, harden=True, attack_type="None")

	# Generate attack data in batches
	perturbed_X = np.array([])
	for i in range(0, attack_X.shape[0], FLAGS.batch_size):
		mini_batch = attack_X[i: i+FLAGS.batch_size,:]
		if mini_batch.shape[0] == 0:
			break
		adv_x_mini = attack.generate_np(mini_batch, **attack_params)
		if perturbed_X.shape[0] != 0:
			perturbed_X = np.append(perturbed_X, adv_x_mini, axis=0)
		else:
			perturbed_X = adv_x_mini

	# Calculate attack success rate (1 - classification rate)
	fooled_rate = 1 - model.evaluate(perturbed_X, attack_Y, batch_size=FLAGS.batch_size)[1]
	print("\nError on adversarial examples: %f" % (fooled_rate))

	# Save examplesi if specified
	if FLAGS.save_here:
		np.save(FLAGS.save_here + "_x.npy", perturbed_X)
		np.save(FLAGS.save_here + "_y.npy", attack_Y)

if __name__ == "__main__":
	app.run()	
