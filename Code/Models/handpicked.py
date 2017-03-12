import common

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils

import numpy as np
import vbow


def modelF(features=10, hidden_neurons=64, nb_classes=10, learning_rate=1.0):
	model = Sequential()
	model.add(Dense(hidden_neurons, input_shape=(features,)))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(hidden_neurons))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(hidden_neurons/2))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy')
	return model
