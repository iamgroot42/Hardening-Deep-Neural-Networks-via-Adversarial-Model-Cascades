import common
import utils_cifar

import numpy as np
from sklearn import svm


def error(model, X_test, Y_test):
	Y_svm = model.predict(X_test)
	numerator = (1*(Y_svm==np.argmax(Y_test,axis=1))).sum()
	acc = numerator / float(Y_test.shape[0])
	return acc


def modelCS(X_train, Y_train, X_test, Y_test):
	clf = svm.SVC(kernel='rbf',verbose=True)
	clf.fit(X_train, np.argmax(Y_train, axis=1))
	print(error(clf, X_test, Y_test))
	return clf


if __name__ == "__main__":
	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
	X_test = X_test.reshape(len(X_test),-1)
	X_train = X_train.reshape(len(X_train),-1)
	print("Reshaped!")
	print(X_train.shape)
	print(Y_train.shape)
	model = modelCS(X_train, Y_train, X_test, Y_test)
