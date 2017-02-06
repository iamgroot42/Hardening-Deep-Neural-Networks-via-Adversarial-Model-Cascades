from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils

import numpy as np
import vbow


def modelF(logits=False,input_ph=None, features=10, hidden_neurons=64, nb_classes=10):
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
	if logits:
		logits_tensor = model(input_ph)
	model.add(Activation('softmax'))
	if logits:
		return model, logits_tensor
	else:
		return model
