import common
import keras
from tensorflow.python.platform import app
from keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from cleverhans.utils_keras import KerasModelWrapper
import data_load, helpers

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_string('model', 'saved_model.h5', 'Path where model is saved')
flags.DEFINE_string('attack_name', 'fgsm', 'Name of attack against which adversarial hardening is to be performed')
flags.DEFINE_string('save_here', None, 'Path to save perturbed examples')
flags.DEFINE_string('mode', 'attack', '(attack/harden/multiple)')
flags.DEFINE_boolean('multiattacks', False, 'Single attack against proxy, or multiple attacks?')

def main(argv=None):
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)
	datagen = dataObject.data_generator()
	atack_X, attack_Y = None, None
	if FLAGS.mode == "harden":
		(attack_X, attack_Y), _ = dataObject.get_blackbox_data()
	elif FLAGS.mode == "attack":
		attack_X, attack_Y = dataObject.get_attack_data()
	else:
		raise Exception("Invalid mode specified!")
		exit()
	n_classes = attack_Y.shape[1]
	if FLAGS.dataset == "cifar10":
		keras.backend.set_image_dim_ordering('th')
		attack_X = attack_X.transpose((0, 3, 1, 2))
	model = load_model(FLAGS.model)
	if not FLAGS.multiattacks:
		attack, attack_params = helpers.get_appropriate_attack(FLAGS.dataset, dataObject.get_range(), FLAGS.attack_name ,KerasModelWrapper(model), common.sess, harden=True, attack_type="black")
		perturbed_X = helpers.performBatchwiseAttack(attack_X, attack, attack_params, FLAGS.batch_size)
	else:
		attacks = FLAGS.attack_name.split(',')
		attacks = attacks[1:]
		attack_params = []
		clever_wrapper = KerasModelWrapper(model)
		for attack in attacks:
			attack_params.append(helpers.get_appropriate_attack(FLAGS.dataset, dataObject.get_range(), attack, clever_wrapper, common.sess, harden=True, attack_type="black"))
		attack_Y_shuffled = []
		perturbed_X = []
		attack_indices = np.array_split(np.random.permutation(len(attack_Y)), len(attacks))
		for i, (at, atp) in enumerate(attack_params):
			adv_data = helpers.performBatchwiseAttack(attack_X[attack_indices[i]], at, atp, FLAGS.batch_size)
			perturbed_X.append(adv_data)
			attack_Y_shuffled.append(attack_Y[attack_indices[i]])
		perturbed_X, attack_Y = np.vstack(perturbed_X), np.vstack(attack_Y)
	fooled_rate = 1 - model.evaluate(perturbed_X, attack_Y, batch_size=FLAGS.batch_size)[1]
	print("\nError on adversarial examples: %f" % (fooled_rate))
	if FLAGS.dataset == "cifar10":
		perturbed_X = perturbed_X.transpose((0, 2, 3, 1))
	if FLAGS.save_here:
		np.save(FLAGS.save_here + "_x.npy", perturbed_X)
		np.save(FLAGS.save_here + "_y.npy", attack_Y)

if __name__ == "__main__":
	app.run()