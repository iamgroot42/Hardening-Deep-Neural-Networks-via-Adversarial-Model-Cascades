import common

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


def placeholder_shapes():
	return (None, 3, 32, 32), (None, 10)


def placeholder_shapes_handpicked(K):
	return (None, K), (None, 10)


def data_cifar():
	img_rows = 32
	img_cols = 32
	nb_classes = 10
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return X_train, Y_train, X_test, Y_test


def data_cifar_raw():
	img_rows = 32
	img_cols = 32
	nb_classes = 10
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return X_train, Y_train, X_test, Y_test
