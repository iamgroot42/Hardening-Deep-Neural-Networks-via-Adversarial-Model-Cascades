from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.utils import np_utils


def placeholder_shapes():
	return (None, 1, 28, 28), (None, 10)


def placeholder_shapes_flat():
	return (None, 28 * 28), (None, 10)


def data_mnist():
	# These values are specific to MNIST
	img_rows = 28
	img_cols = 28
	nb_classes = 10
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return X_train, Y_train, X_test, Y_test


def modelA(logits=False,input_ph=None, img_rows=28, img_cols=28, nb_filters=64, nb_classes=10):
	model = Sequential()
	model.add(Dropout(0.2, input_shape=(3, img_rows, img_cols)))
	model.add(Convolution2D(nb_filters, 8, 8,subsample=(2, 2),border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters * 2, 6, 6, subsample=(2, 2),border_mode="valid"))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters *2, 5, 5, subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(nb_classes))
	if logits:
		logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	if logits:
		return model, logits_tensor
	else:
		return model


def modelB(logits=False,input_ph=None, img_rows=28, img_cols=28, nb_filters=64, nb_classes=10):
	model = Sequential()
	model.add(Convolution2D(nb_filters, 3, 3,
			border_mode='valid',
			input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	if logits:
		logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	if logits:
		return model, logits_tensor
	else:
		return model


def modelC(logits=False,input_ph=None, img_rows=28, img_cols=28, hidden_neurons=512, nb_classes=10):
	model = Sequential()
	model.add(Dense(hidden_neurons, input_shape=(img_rows * img_cols,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(hidden_neurons))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	if logits:
		logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	if logits:
		return model, logits_tensor
	else:
		return model


def model_atrous(logits=False,input_ph=None, img_rows=28, img_cols=28, nb_filters=64, nb_classes=10):
	model = Sequential()
	model.add(AtrousConvolution2D(nb_filters, 3, 3,
		border_mode='valid',
		input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(AtrousConvolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	if logits:
		logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	if logits:
		return model, logits_tensor
	else:
		return model


def model_separable(logits=False,input_ph=None, img_rows=28, img_cols=28, nb_filters=64, nb_classes=10):
	model = Sequential()
	model.add(SeparableConvolution2D(nb_filters, 3, 3,
		border_mode='valid',
		input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(SeparableConvolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	if logits:
		logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	if logits:
		return model, logits_tensor
	else:
		return model
