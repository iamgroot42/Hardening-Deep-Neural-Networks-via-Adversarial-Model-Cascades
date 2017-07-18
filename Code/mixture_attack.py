import os
import common

import tensorflow as tf

import keras
from keras.models import model_from_json, load_model
from keras.utils import np_utils
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import utils_mnist, utils_cifar
from utils_tf import batch_eval
import utils
import helpers

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('fgsm_eps', 0.1, 'Tunable parameter for FGSM')
flags.DEFINE_string('models_directory', './', 'Path to directory containing proxy models to be used')
flags.DEFINE_string('models_data_directory', './', 'Path to Directory where Union, Intersection and individual adversarials will be stored')


def intersect(a, b):
	return list(set(a) & set(b))


def union(a, b):
	return list(set(a) | set(b))


class NormHashNumpy:
	def __init__(self, obj, y):
		self.obj = obj
		self.y = y
		self.norm = int(np.linalg.norm(obj) * 100)
	def __eq__(self, other):
		return self.norm == other.norm
	def __hash__(self):
		return self.norm


def main(argv=None):
	nb_classes = 100
	tf.set_random_seed(1234)
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)
	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
	x_shape, y_shape = utils_cifar.placeholder_shapes()
	x = tf.placeholder(tf.float32, shape=x_shape)
	y = tf.placeholder(tf.float32, shape=y_shape)

	# Load all models
	models = []
	for model in os.listdir(FLAGS.models_directory):
		models.append([model, load_model(FLAGS.models_directory + model)])
	X_test_bm, Y_test_bm, X_test_pm, Y_test_pm = helpers.jbda(X_test, Y_test, prefix="adv", n_points=100, nb_classes=nb_classes)
	perturbed_data = []
	for model in models:
		predictions = model[1](x)
		adv_x = helpers.fgsm(x, predictions, eps=FLAGS.fgsm_eps)
		X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test_pm])
		accuracy = model[1].evaluate(X_test_adv, Y_test_pm, batch_size=FLAGS.batch_size)
		print('\nMisclassification accuracy for %s on adversarial examples: %s'%( model[0] ,str((1.0 - accuracy[1])*100)))
		perturbed_data.append([X_test_adv, Y_test_pm])

	# Create union and intersection sets
	original_shape = perturbed_data[0][0].shape
	count = original_shape[0]
	original_shape = original_shape[1:]
	uni, inter = np.reshape(perturbed_data[0][0], (count, np.prod(original_shape))), np.reshape(perturbed_data[0][0], (count, np.prod(original_shape)))
	UNI = []
	INTER = []
	for i in range(len(uni)):
		UNI.append(NormHashNumpy(uni[i], perturbed_data[0][1][i]))
		INTER.append(NormHashNumpy(uni[i], perturbed_data[0][1][i]))
	uni = UNI
	inter = INTER
	for other_data in perturbed_data[1:]:
		reshaped_otherdata = np.reshape(other_data[0], (count, np.prod(original_shape)))
		ROD = []
		for i in range(len(reshaped_otherdata)):
			ROD.append(NormHashNumpy(reshaped_otherdata[i], other_data[1][i]))
		reshaped_otherdata = ROD
		uni = union(uni, reshaped_otherdata)
		inter = intersect(inter, reshaped_otherdata)
	print "%s elements in union" % (len(uni))
	print "%s elements in intersection" % (len(inter))

	#Reshape union and intersection into normal shape
	uni_y = np.array([x.y for x in uni])
	inter_y = np.array([x.y for x in inter])
	uni = np.array([np.reshape(x.obj, original_shape) for x in uni])
	inter = np.array([np.reshape(x.obj, original_shape) for x in inter])
	#Save adversarial inputs
	for i in range(len(models)):
		np.save(FLAGS.models_data_directory + models[i][0], np.reshape(perturbed_data[i][0], (count, original_shape[0], original_shape[1], original_shape[2])))
	np.save(FLAGS.models_data_directory + "union", uni)
	np.save(FLAGS.models_data_directory + "intersection", inter)
	np.save(FLAGS.models_data_directory + "labels", Y_test_pm)
	np.save(FLAGS.models_data_directory + "labels_union", uni_y)
	np.save(FLAGS.models_data_directory + "labels_intersection", inter_y)


if __name__ == '__main__':
	app.run()
