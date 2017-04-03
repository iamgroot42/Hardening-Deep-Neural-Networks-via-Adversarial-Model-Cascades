import common

import tensorflow as tf
import numpy as np
import keras
from os.path import exists as file_exists
import json

from tensorflow.python.platform import app
from keras.models import load_model
from tensorflow.python.platform import flags

import utils_mnist, utils_cifar
from Models import autoencoder, handpicked, nn_svm, vbow, cnn, sota
import helpers

from sklearn.cluster import KMeans
from sklearn.externals import joblib

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_integer('num_clusters', 10, 'Number of clusters in vbow')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('save_here', 'saved_model', 'Path where model is to be saved')
flags.DEFINE_string('cluster', 'C.pkl', 'Path where cluster/SVM model is to be saved')
flags.DEFINE_boolean('is_blackbox', False , 'Whether the model is the blackbox model, or the proxy model')
flags.DEFINE_integer('is_autoencoder', 0 , 'Whether the model involves an autoencoder(1), handpicked features(2), \
 a CNN with an attached SVM(3), or none(0)')
flags.DEFINE_string('arch', 'arch.json', 'Path where cluster/SVM model is to be saved')
flags.DEFINE_integer('per_class_adv', 100 , 'Number of adversarial examples to be picked per class')
flags.DEFINE_string('proxy_x', 'PX.npy', 'Path where proxy training data is to be saved')
flags.DEFINE_string('proxy_y', 'PY.npy', 'Path where proxy training data labels are to be saved')
flags.DEFINE_string('specialCNN', 'normal', 'if the CNN to be used should be state-of-the-art, normal, have atrous or separable')
flags.DEFINE_boolean('retraining',False, 'if the CNN is being finetuned')


def main(argv=None):
	n_classes=100
	if FLAGS.is_blackbox:
		print("Starting to train blackbox model")
	else:
		print("Starting to train proxy model")
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	sess = tf.Session()
	keras.backend.set_session(sess)

	if FLAGS.is_autoencoder == 2 and FLAGS.is_blackbox:
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar_raw()
		X_train_bm, Y_train_bm, X_train_pm, Y_train_pm = helpers.jbda(X_train, Y_train, "train", 500, n_classes)
		X_train, Y_train = X_train_bm, Y_train_bm
	else:
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
		X_train_bm, Y_train_bm, X_train_pm, Y_train_pm = helpers.jbda(X_train, Y_train, "train", 500, n_classes)
		X_train, Y_train = X_train_bm, Y_train_bm


	label_smooth = .1
	Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

	if FLAGS.is_blackbox:
		X_train_p, Y_train_p = X_train, Y_train
	else:
		X_train_p = np.load(FLAGS.proxy_x)
		Y_train_p = np.load(FLAGS.proxy_y)

	# Black-box network
	if FLAGS.is_blackbox:
		if FLAGS.is_autoencoder != 3:
			if FLAGS.is_autoencoder == 0:
				if FLAGS.retraining:
					# Retraining using adversarial data
					adv_x = np.load(FLAGS.proxy_x)
					adv_y = np.load(FLAGS.proxy_y)
					X_train_p = np.concatenate((X_train, adv_x))
					Y_train_p = np.concatenate((Y_train, adv_y))
					model = load_model(FLAGS.save_here)
				elif file_exists(FLAGS.save_here):
					print "Cached BlackBox model found"
					return
				if FLAGS.specialCNN == 'atrous':
					model = cnn.model_atrous(img_rows=32,img_cols=32,nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
				elif FLAGS.specialCNN == 'separable':
					model = cnn.model_separable(img_rows=32,img_cols=32,nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
				elif FLAGS.specialCNN == 'sota':
					model = sota.cnn_cifar100(FLAGS.learning_rate)
				else:
					model = cnn.modelB(img_rows=32,img_cols=32,nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
			elif FLAGS.is_autoencoder == 1:
				if file_exists(FLAGS.save_here):
					print "Cached BlackBox model found"
					return
				model = autoencoder.modelD(X_train_p, X_test, ne=FLAGS.nb_epochs, bs=FLAGS.batch_size, nb_classes=n_classes, learning_rate=FLAGS.learnig_rate)
			elif FLAGS.is_autoencoder == 2:
				if file_exists(FLAGS.save_here):
					print "Cached BlackBox model found"
					return
				clustering = KMeans(n_clusters=FLAGS.num_clusters, random_state=0, max_iter=100, verbose=1, n_init=3)
				X_train_p, clustering =  vbow.cluster_features(X_train_p, clustering)
				joblib.dump(clustering, FLAGS.cluster)
				X_test = vbow.img_to_vect(X_test, clustering)
				model = handpicked.modelF(features=FLAGS.num_clusters,nb_classes=n_classes)
			datagen = utils_cifar.augmented_data(X_train_p)
			X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
			model.fit_generator(datagen.flow(X_tr, y_tr,
				batch_size=FLAGS.batch_size),
				steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
				epochs=FLAGS.nb_epochs,
				validation_data=(X_val, y_val))
			accuracy = model.evaluate(X_test, Y_test, batch_size=FLAGS.batch_size)
			print('\nTest accuracy for black-box model: ' + str(accuracy[1]*100))
			model.save(FLAGS.save_here)
		else:
			datagen = utils_cifar.augmented_data(X_train_p)
			X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
			if file_exists(FLAGS.save_here):
				print "Cached BlackBox model found"
				model = load_model(FLAGS.save_here)
			else:
				model = cnn_cifar100(FLAGS.learning_rate)
				model.fit_generator(datagen.flow(X_tr, y_tr,
					batch_size=FLAGS.batch_size),
					steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
					epochs=FLAGS.nb_epochs,
					validation_data=(X_val, y_val))
			NN, SVM = nn_svm.modelCS(model, datagen, X_tr, y_tr, X_val,y_val)
			acc = nn_svm.hybrid_error(X_test, Y_test, NN, SVM)
			print('\nTest accuracy for black-box model: ' + str(acc[1]*100))
			NN.save(FLAGS.save_here)
			joblib.dump(SVM, FLAGS.cluster)
			with open(FLAGS.arch, 'w') as outfile:
				json.dump(NN.to_json(), outfile)
	# Proxy network
	else:
		if file_exists(FLAGS.save_here):
			print "Caches proxy found"
			return
		model = cnn.modelA(nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
		datagen = utils_cifar.augmented_data(X_train_p)
		X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
		model.fit_generator(datagen.flow(X_tr, y_tr,
			batch_size=FLAGS.batch_size),
			steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
			epochs=FLAGS.nb_epochs,
			validation_data=(X_val, y_val))
		accuracy = model.evaluate(X_test, Y_test, batch_size=FLAGS.batch_size)
		print('\nTest accuracy for proxy model: ' + str(accuracy[1]*100))
		model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
