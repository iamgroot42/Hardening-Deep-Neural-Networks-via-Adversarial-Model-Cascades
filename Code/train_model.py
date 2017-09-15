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
from Models import autoencoder, nn_svm, cnn, sota, nn
import helpers

from sklearn.cluster import KMeans
from sklearn.externals import joblib

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
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
flags.DEFINE_integer('proxy_level', 1, 'Model with scale (1,2,4)')
flags.DEFINE_boolean('retraining', False, 'Whether the model is being retrained or traned from scratch')

def main(argv=None):
	n_classes=100
	print("Starting to train model")
	tf.set_random_seed(1234)
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')
	# Create TF session and set as Keras backend session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	if FLAGS.is_blackbox:
		X, Y, _, _ = utils_cifar.data_cifar()
		X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, n_classes)
		indices = np.random.choice(len(Y_train_p), int(len(Y_train_p)/2))
		X_train_p = X_train_p[indices]
		Y_train_p = Y_train_p[indices]
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
					X_train_p = np.concatenate((X_train_p, adv_x))
					Y_train_p = np.concatenate((Y_train_p, adv_y))
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
				model = autoencoder.modelD(X_train_p, X_test, ne=FLAGS.nb_epochs, bs=FLAGS.batch_size, nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
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
			accuracy = model.evaluate(X_val, y_val, batch_size=FLAGS.batch_size)
			print('\nTest accuracy for black-box model: ' + str(accuracy[1]*100))
			model.save(FLAGS.save_here)
		else:
			datagen = utils_cifar.augmented_data(X_train_p)
			X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
			if file_exists(FLAGS.save_here) and file_exists(FLAGS.cluster):
				print "Cached modified blackbox and SVM found"
				return
			elif file_exists(FLAGS.save_here):
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
			acc = nn_svm.hybrid_error(X_val, y_val, NN, SVM)
			print('\nTest accuracy for black-box model: ' + str(acc*100))
			NN.save(FLAGS.save_here)
			joblib.dump(SVM, FLAGS.cluster)
			with open(FLAGS.arch, 'w') as outfile:
				json.dump(NN.to_json(), outfile)
	# Proxy network
	else:
		if FLAGS.proxy_level == 1:
			model = cnn.modelA(nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
		elif FLAGS.proxy_level == 2:
			model = cnn.modelA2(nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
		else:
			model = cnn.modelA4(nb_classes=n_classes, learning_rate=FLAGS.learning_rate)
		datagen = utils_cifar.augmented_data(X_train_p)
		X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
		model.fit_generator(datagen.flow(X_tr, y_tr,
			batch_size=FLAGS.batch_size),
			steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
			epochs=FLAGS.nb_epochs,
			validation_data=(X_val, y_val))
		accuracy = model.evaluate(X_val, y_val, batch_size=FLAGS.batch_size)
		print('\nTest accuracy for proxy model: ' + str(accuracy[1]*100))
		model.save(FLAGS.save_here)

if __name__ == '__main__':
	app.run()

