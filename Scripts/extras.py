import sys

from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta

import numpy as np
from sklearn import svm
from keras.models import load_model

from sklearn.externals import joblib

import keras

def MLP(model, learning_rate):
	interm_l = Model(input=model.input, output=model.layers[6].output)
	hidden_neurons = 256
	interm_l = Sequential()
	interm_l.add(encoder)
	interm_l.add(Flatten())
	interm_l.add(Dense(hidden_neurons))
	interm_l.add(Activation('relu'))
	interm_l.add(Dropout(0.5))
	interm_l.add(Dense(hidden_neurons/2))
	interm_l.add(Activation('relu'))
	interm_l.add(Dropout(0.4))
	interm_l.add(Dense(hidden_neurons/4))
	interm_l.add(Activation('relu'))
	interm_l.add(Dropout(0.3))
	interm_l.add(Dense(100))
	interm_l.add(Activation('softmax'))
	interm_l.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return interm_l


def get_output(X_test, model, cluster):
	out = model.layers[6].output
        out = Flatten()(out)
        flat_model = Model(input=model.input, output=out)
	X_test_SVM = flat_model.predict(X_test)
	Y_svm = cluster.predict(X_test_SVM)
	return Y_svm


def hybrid_error(X_test, Y_test, model, cluster):
	Y_svm = get_output(X_test, model, cluster)
	numerator = (1*(Y_svm==np.argmax(Y_test,axis=1))).sum()
	acc = numerator / float(Y_test.shape[0])
	return acc


def SVM(model, X_tr, y_tr, rbf=True):
	out = model.layers[6].output
	out = Flatten()(out)
	interm_l = Model(input=model.input, output=out)
	X_train_SVM = interm_l.predict(X_tr)
	if rbf:
		clf = svm.SVC(kernel='rbf', probability=True)
	else:
		clf = svm.SVC(kernel='linear', probability=True)
	clf.fit(X_train_SVM, np.argmax(y_tr, axis=1))
	return clf


def validation_split(X, y, validation_split=0.2):
	num_points = len(X)
	validation_indices = np.random.choice(num_points, int(num_points * validation_split))
	train_indices = list(set(range(num_points)) - set(validation_indices))
	X_train, y_train = X[train_indices], y[train_indices]
	X_val, y_val = X[validation_indices], y[validation_indices]
	return X_train, y_train, X_val, y_val


def augmented_data(X):
	datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        data_format="channels_first") # (channel, row, col) format per image
	datagen.fit(X)
    	return datagen


def data_cifar(normalize=True):
	img_rows = 32
	img_cols = 32
	nb_classes = 100
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = cifar100.load_data()
	X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	if normalize:
		X_train /= 255
		X_test /= 255
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
	X = np.load("PX.npy")
	Y = np.load("PY.npy")
	print X.shape, Y.shape
	encoder = load_model("AEC")
	xtr, ytr, xval, yval = validation_split(X, Y)
	_, _, X_test, Y_test = data_cifar()
	learning_rate = float(sys.argv[1])
	nb_epochs = int(sys.argv[2])
	# MLP
	# model = MLP(encoder, learning_rate)
	# datagen = augmented_data(xtr)
	#model.fit_generator(datagen.flow(xtr, ytr,
	#	batch_size=16),
	#	steps_per_epoch=xtr.shape[0] // 16,
	#	epochs=nb_epochs,
	#	validation_data=(xval, yval))
	#accuracy = model.evaluate(X_test, Y_test, batch_size=16)
	#print('\nTest accuracy for model: ' + str(accuracy[1]*100))
	#model.save(sys.argv[3])
	# SVM
	svm = SVM(encoder, X, Y, False)
	joblib.dump(svm, "svm.pkl")
	acc = hybrid_error(X_test, Y_test, encoder, svm)
	print('\nTest accuracy for model: ' + str(acc*100))
	joblib.dump(svm, "svm.pkl")
	with open(FLAGS.arch, 'w') as outfile:
		json.dump(NN.to_json(), outfile)

