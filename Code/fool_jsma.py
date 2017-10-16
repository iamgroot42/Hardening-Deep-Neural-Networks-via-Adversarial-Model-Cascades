import common

import keras
import numpy as np

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import batch_eval
import utils_cifar, utils_mnist, utils_svhn
import helpers

from keras.models import load_model

FLAGS = flags.FLAGS


flags.DEFINE_integer('n_subset_classes', 10, 'Number of target classes')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')

import os
from six.moves import xrange

from attacks import SaliencyMapMethod

def main(argv=None):
	n_classes = 10
	image_shape = (32, 32, 3)
	if FLAGS.dataset == 'cifar100':
		n_classes = 100
		_, _, X_test, Y_test = utils_cifar.data_cifar()
		x_shape, y_shape = utils_cifar.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=10, nb_classes=n_classes)
	elif FLAGS.dataset == 'mnist':
		_, _, X_test, Y_test = utils_mnist.data_mnist()
		x_shape, y_shape = utils_mnist.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=n_classes)
		image_shape = (28, 28, 1)
	elif FLAGS.dataset == 'svhn':
		_, _, X_test, Y_test = utils_svhn.data_svhn()
		x_shape, y_shape = utils_svhn.placeholder_shapes()
		X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=n_classes)
	else:
		print "Invalid dataset. Exiting."
		exit()

	keras.layers.core.K.set_learning_phase(0)

	tf.set_random_seed(1234)
	if keras.backend.image_dim_ordering() != 'tf':
		keras.backend.set_image_dim_ordering('tf')

	#Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)

	model = load_model(FLAGS.model_path)

	source_samples = Y_test_pm.shape[0]
	labels = []
	preds = model(x)

	print('Crafting ' + str(source_samples) + ' * ' + str(n_classes - 1) + ' adversarial examples')

	# Keep track of success (adversarial example classified in target)
	results = np.zeros((n_classes, source_samples), dtype='i')

	# Rate of perturbed features for each test set example and target class
	perturbations = np.zeros((n_classes, source_samples), dtype='f')

	# Define the SaliencyMapMethod attack object
	jsma = SaliencyMapMethod(model, back='tf', sess=sess)

	generated_images = []

	# Loop over the samples we want to perturb into adversarial examples
	for sample_ind in xrange(0, source_samples):
		print('Attacking input %i/%i' % (sample_ind + 1, source_samples))

		# We want to find an adversarial example for each possible target class
		# (i.e. all classes that differ from the label given in the dataset)
		current_class = int(np.argmax(Y_test_pm[sample_ind]))
		target_classes = helpers.other_classes(n_classes, current_class)

		# Reduce search space for target classes to increase speed
		if len(target_classes) > FLAGS.n_subset_classes:
			target_classes = np.random.choice(target_classes, FLAGS.n_subset_classes, replace=False)

		# Loop over all target classes
		for target in target_classes:
			print('Generating adv. example for target class %i' % target)
			try:

				# This call runs the Jacobian-based saliency map approach
				one_hot_target = np.zeros((1, n_classes), dtype=np.float32)
				one_hot_target[0, target] = 1
				jsma_params = {'theta': 1., 'gamma': 0.1,
							   'nb_classes': n_classes, 'clip_min': 0.,
							   'clip_max': 1., 'targets': y,
							   'y_val': one_hot_target}
				if FLAGS.dataset =='svhn':
					jsma_params['clip_min'] = 0
					jsma_params['clip_max'] = 255

				adv_x = jsma.generate_np(X_test_pm[sample_ind:(sample_ind+1)],
										 **jsma_params)

				generated_images.append(adv_x[0])
				labels.append(Y_test_pm[sample_ind])

				# Check if success was achieved
				res = int(helpers.model_argmax(sess, x, preds, adv_x) == target)

				# Computer number of modified features
				adv_x_reshape = adv_x.reshape(-1)
				test_in_reshape = X_test_pm[sample_ind].reshape(-1)
				nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
				percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

				# Update the arrays for later analysis
				results[target, sample_ind] = res
				perturbations[target, sample_ind] = percent_perturb

			except:
				continue

	print('--------------------------------------')

	# Compute the number of adversarial examples that were successfully found
	nb_targets_tried = ((n_classes - 1) * source_samples)
	succ_rate = float(np.sum(results)) / nb_targets_tried
	print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))

	# Compute the average distortion introduced by the algorithm
	percent_perturbed = np.mean(perturbations)
	print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

	# Compute the average distortion introduced for successful samples only
	percent_perturb_succ = np.mean(perturbations * (results == 1))
	print('Avg. rate of perturbed features for successful '
		  'adversarial examples {0:.4f}'.format(percent_perturb_succ))

	# Close TF session
	sess.close()

	# Save adversarial images
	generated_images = np.array(generated_images)
	labels = np.array(labels)

	assert(generated_images.shape[0]==labels.shape[0])

	np.save(FLAGS.adversary_path_y, labels)
	np.save(FLAGS.adversary_path_x, generated_images)


if __name__ == '__main__':
	app.run()


