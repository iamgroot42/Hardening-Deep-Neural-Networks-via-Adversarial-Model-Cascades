import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

tf.set_random_seed(1234)

FLAGS = flags.FLAGS

import keras
from cleverhans.attacks import DeepFool 

from keras_to_ch import KerasModelWrapper
import data_load

flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('iters', 50, 'Maximum iterations')


def main(argv=None):
	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)()

	if dataObject is None:
		print "Invalid dataset; exiting"
		exit()
	
	if FLAGS.mode == 'attack':
		(X, Y) = dataObject.get_attack_data()
	else:
		(X, Y) = dataObject.get_hardening_data()

	if keras.backend.image_dim_ordering() != 'tf':
		keras.backend.set_image_dim_ordering('tf')

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	raw_model = keras.models.load_model(FLAGS.model_path)
	model = KerasModelWrapper(raw_model)

	n_classes = Y.shape[1]
	deepfool = DeepFool(model, sess=sess)
	deepfool.parse_params(clip_min=0.0, clip_max=1.0, nb_candidate=n_classes, max_iter=FLAGS.iters)
	adv_x = deepfool.generate_np(X_test_pm)

	accuracy = raw_model.evaluate(adv_x, Y_test_pm, batch_size=FLAGS.batch_size)
	print('\nError on adversarial examples: ' + str((1.0 - accuracy[1])))

	np.save(FLAGS.adversary_path_x, adv_x)
	np.save(FLAGS.adversary_path_y, Y)


if __name__ == '__main__':
	app.run()

