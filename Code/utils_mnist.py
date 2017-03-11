import common

from keras.datasets import mnist
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
