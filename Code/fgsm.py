import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras
from cleverhans.attacks import FastGradientMethod 

from keras_to_ch import KerasModelWrapper
import data_load

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('fgsm_eps', 0.1, 'Tunable parameter for FGSM')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where original labels are to be saved')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')

# Set seed for reproducability
tf.set_random_seed(42)

def main(argv=None):
	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)()

	if dataObject is None:
		print "Invalid dataset; exiting"
		exit()
	
	if FLAGS.mode == 'attack':# Set seed for reproducability
		(X, Y) = dataObject.get_attack_data()
	else:
		(X, Y) = dataObject.get_hardening_data()

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	raw_model = keras.models.load_model(FLAGS.model_path)
	model = KerasModelWrapper(raw_model)

	madry = attacks.FastGradientMethod(model, sess=sess)
	adv_x = madry.generate_np(X, eps=FLAGS.epsilon, clip_min=0.0, clip_max=1.0)

	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = raw_model.evaluate(adv_x, Y, FLAGS=batch_size=FLAGS.batch_size)
	print('\nError on adversarial examples: ' + str((1.0 - accuracy[1])))

	np.save(FLAGS.adversary_path_x, X)
	np.save(FLAGS.adversary_path_y, Y)


if __name__ == '__main__':
	app.run()
