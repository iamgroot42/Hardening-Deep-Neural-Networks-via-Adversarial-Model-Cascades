import common

import tensorflow as tf
import numpy as np
import keras

from tensorflow.python.platform import app
from keras.models import load_model
from tensorflow.python.platform import flags
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import data_load
from Models import sota

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', '(train,finetune)')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_integer('nb_epochs', 300, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_string('save_here', 'saved_model', 'Path where model is to be saved')
flags.DEFINE_string('proxy_x', 'PX.npy', 'Path where proxy training data is to be saved')
flags.DEFINE_string('proxy_y', 'PY.npy', 'Path where proxy training data labels are to be saved')
flags.DEFINE_float('label_smooth', 0, 'Amount of label smoothening to be applied')


def main(argv=None):
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	# Load dataset
	adv_x = None
	adv_y = None

	if FLAGS.mode == 'finetune':
		adv_x = np.load(FLAGS.proxy_x)
		adv_y = np.load(FLAGS.proxy_y)

	# Initialize data object
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)(adv_x, adv_y)

	if dataObject is None:
		print "Invalid dataset; exiting"
		exit()

	# Black-box network
	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
	X_train, Y_train, X_validation, Y_validation = dataObject.validation_split(blackbox_Xtrain, blackbox_Ytrain, 0.2)

	# Early stopping and dynamic lr
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.001, verbose=1)
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1)

	if FLAGS.label_smooth > 0:
		Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)

	if FLAGS.mode == 'train':
		n_classes = Y_train.shape[1]
		model = sota.get_appropriate_model(FLAGS.dataset)(FLAGS.learning_rate, n_classes)
	else:
		model = load_model(FLAGS.save_here)
		model.optimizer.lr.assign(FLAGS.learning_rate)

	datagen = dataObject.data_generator()
	datagen.fit(X_train)
	model.fit_generator(datagen.flow(X_train, Y_train,
		batch_size=FLAGS.batch_size),
		steps_per_epoch=X_train.shape[0] // FLAGS.batch_size,
		epochs=FLAGS.nb_epochs,
		callbacks=[reduce_lr, early_stop],
		validation_data=(X_validation, Y_validation))

	accuracy = model.evaluate(X_test, Y_test, batch_size=FLAGS.batch_size)
	print('\nTest accuracy for black-box model: ' + str(accuracy[1]))
	model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
