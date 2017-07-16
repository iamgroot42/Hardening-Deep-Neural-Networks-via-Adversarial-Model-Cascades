import common
from sota import cnn_cifar100

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop

import numpy as np
from sklearn import svm


def get_output(X_test, model, cluster):
	X_test_SVM = model.predict(X_test)
	Y_svm = cluster.predict(X_test_SVM)
	return Y_svm


def hybrid_error(X_test, Y_test, model, cluster):
	Y_svm = get_output(X_test, model, cluster)
	numerator = (1*(Y_svm==np.argmax(Y_test,axis=1))).sum()
	acc = numerator / float(Y_test.shape[0])
	return acc


def modelCS(final_model, datagen, X_tr, y_tr, X_val, y_va):
	# Remove last layers to get encoding for SVM
	interm_l = Model(input=final_model.input, output=final_model.layers[-4].output)
	X_train_SVM = interm_l.predict(X_tr)
	clf = svm.SVC(kernel='rbf', probability=True)
	clf.fit(X_train_SVM, np.argmax(y_tr, axis=1))
	return interm_l, clf
