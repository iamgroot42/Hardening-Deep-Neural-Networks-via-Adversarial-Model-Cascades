import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.optimizers import Adadelta, SGD
from keras.regularizers import l2
from keras.initializers import he_normal

import keras


def proxy(n_classes=10, mnist=False, learning_rate=1e-1):
	img_input = (32, 32, 3)
	if mnist == True:
		img_input = (28, 28, 1)

	model = Sequential()
	model.add(Conv2D(32, 3, 3, border_mode='same', W_regularizer=l2(0.0001), init=he_normal(),
			input_shape=img_input))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3), W_regularizer=l2(0.0001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3, 3), border_mode='same', W_regularizer=l2(0.0001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), W_regularizer=l2(0.0001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(512, W_regularizer=l2(0.0001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=SGD(lr=learning_rate),
		metrics=['accuracy'])
	return model
