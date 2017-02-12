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


def internal_model(ne=50, bs=128, learning_rate=0.001):
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(3, 32, 32)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
	return model


def get_output(X_test, model, cluster):
	X_test_SVM = model.predict(X_test)
	Y_svm = cluster.predict(X_test_SVM)
	return Y_svm


def hybrid_error(X_test, Y_test, model, cluster):
	Y_svm = get_output(X_test, model, cluster)
	numerator = (1*(Y_svm==np.argmax(Y_test,axis=1))).sum()
	acc = numerator / float(Y_test.shape[0])
	return acc


def modelCS(X_train, Y_train, X_test, Y_test, input_ph=None, ne=50, bs=128, learning_rate=0.01):
	final_model = internal_model(ne, bs, learning_rate)
	final_model.fit(X_train, Y_train,
				nb_epoch=ne,
				batch_size=bs,
				validation_data=(X_test, Y_test))
	score = final_model.evaluate(X_test, Y_test)
	print("\nNN-only accuracy: " + str(score))
	# Remove last layers to get encoding for SVM
	interm_l = Model(input=final_model.input,
                                 output=final_model.layers[-4].output)
	X_train_SVM = interm_l.predict(X_train)
	X_test_SVM = interm_l.predict(X_test)
	clf = svm.SVC(kernel='rbf')
	clf.fit(X_train_SVM, np.argmax(Y_train, axis=1))
	return interm_l, clf

