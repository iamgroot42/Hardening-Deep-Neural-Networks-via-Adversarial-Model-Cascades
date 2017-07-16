import common

from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta

import numpy as np


def modelD(X_train, X_test, ne=50, bs=128, learning_rate=0.1, nb_classes=10):
	input_img = Input(shape=(3, 32, 32))
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	# at this point the representation is (8, 4, 4) i.e. 128-dimensional
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
	encoder = Model(input=input_img, output=encoded)
	autoencoder = Model(input=input_img, output=decoded)
	# Configure autoencoder
	autoencoder.compile(loss='mean_squared_error',optimizer=Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
, metrics=['accuracy'])
	autoencoder.fit(X_train, X_train,
				nb_epoch=ne,
				batch_size=bs,
				validation_data=(X_test, X_test))
	score = autoencoder.evaluate(X_test, X_test)[1]
	print("\nAutoencoder accuracy: " + str(score))
	autoencoder.save("AUTO")
	exit()
	# Build ultimate model
	for i in encoder.layers:
		i.trainable = False
	hidden_neurons = 512
	final_model = Sequential()
	final_model.add(encoder)
	final_model.add(Flatten())
	final_model.add(Dense(hidden_neurons))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(hidden_neurons))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(hidden_neurons))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(nb_classes))
	final_model.add(Activation('softmax'))
	final_model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return final_model


def modelE(img_rows=32, img_cols=32, nb_filters=64, nb_classes=10, learning_rate=1.0):
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
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def modelF(X_train, X_test, ne=50, bs=16, learning_rate=0.1, nb_classes=100):
	input_img = Input(shape=(3, 32, 32))
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(input_img)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
        # at this point the representation is of shape (16, 5, 5)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(encoded)
        x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
        x = UpSampling2D((2, 2))(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(x)
	x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(3, 5, 5, activation='relu', border_mode='valid')(x)
        encoder = Model(input=input_img, output=encoded)
        autoencoder = Model(input=input_img, output=decoded)
        # Configure autoencoder
        autoencoder.compile(loss='mean_squared_error',optimizer=Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
, metrics=['accuracy'])
        autoencoder.fit(X_train, X_train,
                                nb_epoch=ne,
                                batch_size=bs,
                                validation_data=(X_test, X_test))
        score = autoencoder.evaluate(X_test, X_test)[1]
        print("\nAutoencoder accuracy: " + str(score))

