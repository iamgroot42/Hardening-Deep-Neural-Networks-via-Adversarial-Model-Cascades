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
flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_string('model', 'saved_model.h5', 'Path where model is saved')
flags.DEFINE_string('attack_name', 'fgsm', 'Name of attack against which adversarial hardening is to be performed')
flags.DEFINE_string('save_here', None, 'Path to save perturbed examples')
flags.DEFINE_string('mode', 'attack', '(attack/harden)')

def main(argv=None):
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)
	datagen = dataObject.data_generator()
	atack_X, attack_Y = None, None
	if FLAGS.mode == "harden":
		attack_X, attack_Y = dataObject.get_hardening_data()
	elif FLAGS.mode == "attack":
		attack_X, attack_Y = dataObject.get_attack_data()
	else:
		raise Exception("Invalid mode specified!")
		exit()
	n_classes, model = attack_Y.shape[1], load_model(FLAGS.model)
	attack, attack_params = helpers.get_appropriate_attack(FLAGS.dataset, dataObject.get_range(), FLAGS.attack_name ,KerasModelWrapper(model), common.sess, harden=True, attack_type="None")
	perturbed_X = helpers.performBatchwiseAttack(attack_X, attack, attack_params, FLAGS.batch_size)
	fooled_rate = 1 - model.evaluate(perturbed_X, attack_Y, batch_size=FLAGS.batch_size)[1]
	print("\nError on adversarial examples: %f" % (fooled_rate))
	if FLAGS.save_here:
		np.save(FLAGS.save_here + "_x.npy", perturbed_X)
		np.save(FLAGS.save_here + "_y.npy", attack_Y)

if __name__ == "__main__":
	app.run()