# Fix for Tensorflow import error while using Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import  numpy as np
import utils_cifar


def ready_model(weights_path, learning_rate=0.1):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(365, activation='softmax'))

	model.load_weights(weights_path)
	# Freeze all layers except the last two
	model.pop()
	model.pop()
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
	return model


def process_data(X_train, X_test):
	try:
		# Try loading data if cached
		X_train = np.load("../Data/upscaled_train.npy")
		X_test = np.load("../Data/upscaled_test.npy")
	except:
		x_tr = []
		x_ts = []
		for x in X_train:
			x_tr.append(cv2.resize(image, (0,0), fx=7, fy=7))
		for x in X_test:
			x_ts.append(cv2.resize(image, (0,0), fx=7, fy=7))
		x_tr = np.array(x_tr)
		x_ts = np.array(x_ts)
		np.save("../Data/upscaled_train.npy", x_tr)
		np.save("../Data/upscaled_test.npy", x_ts)
	return X_train, X_test


def modelFinetuneVGG(X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
	model = ready_model('Data/Pretrained/vgg.h5')
	model.fit(X_train,y_train, nb_epoch=epochs, batch_size=batch_size)
	loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
	print loss_and_metrics
	return model


if __name__ == "__main__":
	X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar()
	X_train, X_test = process_data(X_train, X_test)
	# model = modelFinetuneVGG(X_train, Y_train, X_test, Y_test)
