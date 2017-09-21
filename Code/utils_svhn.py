import common

from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def placeholder_shapes():
	return (None, 3, 32, 32), (None, 10)


def data_svhn():
	# the data, shuffled and split between train and test sets
	X_train, Y_train = np.load("../Code/SVHN/SVHNx_tr.npy"), np.load("../Code/SVHN/SVHNy_tr.npy")
	X_test, Y_test = np.load("../Code/SVHN/SVHNx_te.npy"), np.load("../Code/SVHN/SVHNy_te.npy")
	return X_train, Y_train, X_test, Y_test


def augmented_data(X):
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False,  # randomly flip images
		data_format="channels_first") # (channel, row, col) format per image
	datagen.fit(X)
	return datagen
