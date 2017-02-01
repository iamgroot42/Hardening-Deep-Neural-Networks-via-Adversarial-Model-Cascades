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
from helpers import pop_layer


def modelCS(X_train, Y_train, X_test, Y_test, input_ph=None, ne=1, bs=128, learning_rate=0.2):
	final_model = Sequential()
	final_model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(3, 32, 32)))
	final_model.add(MaxPooling2D((2, 2), border_mode='same'))
	final_model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same'))
	final_model.add(MaxPooling2D((2, 2), border_mode='same'))
	final_model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same'))
	final_model.add(MaxPooling2D((2, 2), border_mode='same'))
	final_model.add(Flatten())
	hidden_neurons = 512
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
	final_model.compile(loss='binary_crossentropy',optimizer='Adadelta')
	before = final_model.to_json()
	final_model.fit(X_train, Y_train,
				nb_epoch=ne,
				batch_size=bs,
				validation_data=(X_test, Y_test))
	score = final_model.evaluate(X_test, Y_test)
	print("\nNN-only accuracy: " + str(score))
	# Remove last layers to get encoding for SVM
	final_model.pop()
	final_model.pop()
	final_model.pop()
	after = final_model.to_json()
	X_train_SVM = final_model.predict(X_train)
	X_test_SVM = final_model.predict(X_test)
	#clf = svm.SVC(kernel='rbf')
	#clf.fit(X_train_SVM, np.argmax(Y_train, axis=1))
	import json
	with open('before', 'w') as outfile:
		json.dump(before, outfile)
	with open('after','w') as outfile:
		json.dump(after, outfile)
	final_model.save("ohhok")
	return final_model, clf


if __name__ == "__main__":
	import utils_cifar
	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
	Y_fin = np_utils.to_categorical(modelCS(X_train, Y_train, X_test, Y_test))
	acc =  100 * np.multiply(Y_test, Y_fin).sum() / Y_test.shape[0]
	print("testing accuracy " + str(acc))
