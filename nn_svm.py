from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils

import numpy as np
from sklearn import svm


def modelD(X_train, Y_train, X_test, Y_test, logits=False,input_ph=None, ne=1, bs=128, learning_rate=0.2):
	input_img = Input(shape=(3, 32, 32))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	encoder = Model(input=input_img, output=encoded)
	hidden_neurons = 512
	final_model = Sequential()
	final_model.add(encoder)
	final_model.add(Flatten())
	final_model.add(Dense(hidden_neurons))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(hidden_neurons/2))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(hidden_neurons/4))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.2))
	final_model.add(Dense(10))
	final_model.add(Activation('softmax'))
	if logits:
		logits_tensor = final_model(input_ph)
	final_model.compile(loss='binary_crossentropy',optimizer='Adadelta')
	final_model.fit(X_train, Y_train,
				nb_epoch=ne,
				batch_size=bs,
				validation_data=(X_test, Y_test))
	score = autoencoder.evaluate(X_test, Y_test)
	print("NN-only accuracy: " + str(score))
	# Remove last layers to get encoding for SVM
	final_model.pop()
	final_model.pop()
	X_train_SVM = final_model.fit(X_train)
	X_test_SVM = final_model.fit(X_test)
	clf = svm.SVC(kernel='rbf')
	clf.fit(X_test_SVM, Y_train)
	print clf.predict(X_test_SVM[0])


if __name__ == "__main__":
	import utils_cifar
	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
	modelD(X_train, Y_train, X_test, Y_test)
