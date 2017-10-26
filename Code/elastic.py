import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras
from cleverhans.attacks import ElasticNetMethod

from keras_to_ch import KerasModelWrapper
import data_load

# Set seed for reproducability
tf.set_random_seed(42)

FLAGS = flags.FLAG

flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_float('beta', 1e-3, 'Value of Beta')


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

	keras.layers.core.K.set_learning_phase(0)

	if keras.backend.image_dim_ordering() != 'tf':
		keras.backend.set_image_dim_ordering('tf')

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	raw_model = keras.models.load_model(FLAGS.model_path)
	model = KerasModelWrapper(raw_model)

	enet = ElasticNetMethod(model, sess=sess)
	x_adv = self.attack.generate_np(x_val, clip_min=0.0, clip_max=1.0, batch_size=FLAGS.batch_size)

	accuracy = raw_model.evaluate(adv_x, Y_test_pm, batch_size = FLAGS.batch_size)
	print('\Error on adversarial examples: ' + str((1.0 - accuracy[1])))

	np.save(FLAGS.adversary_path_x, adv_x)
	np.save(FLAGS.adversary_path_y, Y)


if __name__ == '__main__':
	app.run()
