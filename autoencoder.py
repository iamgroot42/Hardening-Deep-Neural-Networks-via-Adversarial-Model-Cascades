from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils

import numpy as np


def modelD(X_train, X_test, logits=False,input_ph=None, ne=3, bs=128, learning_rate=0.2):
	input_img = Input(shape=(3, 32, 32))
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
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
	encoder = Model(input=input_img, output=encoded)
	autoencoder = Model(input=input_img, output=decoded)
	# Configure autoencoder
	autoencoder.compile(loss='binary_crossentropy',optimizer='Adadelta')
	autoencoder.fit(X_train, X_train,
				nb_epoch=ne,
				batch_size=bs,
				validation_data=(X_test, X_test))
	score = autoencoder.evaluate(X_test, X_test)
	print("\nAutoencoder accuracy: " + str(score))
	# Build ultimate model
	for i in encoder.layers:
		i.trainable = False
	hidden_neurons = 512
	y = Dense(hidden_neurons)(encoder)
	y = Activation('relu')(y)
	y = Dropout(0.2)(y)
	y = Dense(hidden_neurons)(y)
	y = Activation('relu')(y)
	y = Dropout(0.2)(y)
	y = Dense(hidden_neurons)(y)
	y = Activation('relu')(y)
	y = Dropout(0.2)(y)
	y = Dense(10)(y)
	compressed_learner = Activation('softmax')(y)
	final_model = Model(input=input_img, output=compressed_learner)
	if logits:
		logits_tensor = final_model(input_ph)
	final_model.add(Activation('softmax'))
	if logits:
		return final_model, logits_tensor
	else:
		return final_model


def modelD(logits=False,input_ph=None, compressed_shape=(8,4,4), hidden_neurons=512, nb_classes=10):
	model = Sequential()
	model.add(Dense(hidden_neurons, input_shape=(np.prod(compressed_shape),)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(hidden_neurons))
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


def modelE(logits=False,input_ph=None, img_rows=32, img_cols=32, nb_filters=64, nb_classes=10):
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
