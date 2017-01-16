from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np


def learn_encoding(X_train, X_test, ne=10, bs=128, learning_rate=0.1):
	input_img = Input(shape=((3, img_rows, img_cols),))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	# at this point the representation is (8, 4, 4) i.e. 128-dimensional
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
	encoder = Model(input=input_img, output=encoded)
	autoencoder = Model(input=input_img, output=decoded)
	# Configure autoencoder
	autoencoder.compile(loss='binary_crossentropy',optimizer='Adadelta')
	autoencoder.fit(X_train, X_train,
				nb_epoch=ne,
				batch_size=bs,
				validation_data=(X_test, X_test))
	loss_and_metrics = model.evaluate(X_test_comp, y_test, batch_size=bs)
	print(loss_and_metrics)
	return encoder


def modelD(logits=False,input_ph=None, compressed_shape=(8,4,4), hidden_neurons=512, nb_classes=10):
	model = Sequential()
	model.add(Dense(hidden_neurons, input_shape=(np.prod(),)))
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
