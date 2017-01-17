from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


def placeholder_shapes():
	return (None, 3, 32, 32), (None, 10)


def data_cifar():
	# These values are specific to CIFAR
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


def modelD(logits=False,input_ph=None, img_rows=32, img_cols=32, nb_filters=64, nb_classes=10):
	model = Sequential()
	model.add(Convolution2D(nb_filters/2, 3, 3, border_mode='same',
	                        input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters/2, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
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


def learn_encoding(learning_rate, aune, ne, bs, autoencoder_weight_file):
	input_img = Input(shape=((3, img_rows, img_cols),))
	encoded = Dense(100, activation='sigmoid')(input_img)
	decoded = Dense((3, img_rows, img_cols), activation='sigmoid')(encoded)
	autoencoder = Model(input=input_img, output=decoded)
	# Create encoder
	encoder = Model(input=input_img, output=encoded)
	# Configure autoencoder
	autoencoder.compile(loss='binary_crossentropy',optimizer='Adadelta')
	autoencoder.fit(X_train, X_train,
				nb_epoch=aune,
				batch_size=bs,
				validation_data=(X_test, X_test))
	return autoencoder
