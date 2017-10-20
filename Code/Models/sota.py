import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adadelta, SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.initializers import he_normal


def conv_stack(filters, side, activation, model, input_shape=None):
	if not input_shape:
		model.add(Conv2D(filters, (side, side), border_mode='same', W_regularizer=l2(0.01), init=he_normal()))
	else:
		model.add(Conv2D(filters, (side, side), border_mode='same', input_shape=input_shape, W_regularizer=l2(0.01), init=he_normal()))
	model.add(BatchNormalization())
	model.add(activation())


def cifar_svhn(learning_rate, n_classes=100):
	model = Sequential()
	conv_stack(192, 5, ELU, model,(3, 32, 32))
	model.add(MaxPooling2D())

	conv_stack(192, 1, ELU, model)
	conv_stack(240, 3, ELU, model)
	model.add(Dropout(0.1))
	model.add(MaxPooling2D())

	conv_stack(240, 1, ELU, model)
	conv_stack(260, 2, ELU, model)
	model.add(Dropout(0.2))
	model.add(MaxPooling2D())

	conv_stack(260, 1, ELU, model)
	conv_stack(280, 2, ELU, model)
	model.add(Dropout(0.2))
	model.add(MaxPooling2D())

	conv_stack(280, 1, ELU, model)
	conv_stack(300, 2, ELU, model)
	model.add(Dropout(0.2))
	model.add(MaxPooling2D())

	model.add(Flatten())
	model.add(BatchNormalization())

	model.add(Dropout(0.5))
	model.add(Dense(300, W_regularizer=l2(0.01), init=he_normal()))
	model.add(ELU())
	model.add(BatchNormalization())
	model.add(Dense(n_classes, W_regularizer=l2(0.01), init=he_normal()))
	model.add(Activation('softmax'))
	model.compile(optimizer=SGD(lr=learning_rate,momentum=0.9),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def mnist(learning_rate, n_classes=10):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
				activation='relu',
				input_shape=(1,28,28)))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10))
		model.add(Activation('softmax'))
		model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=Adadelta(lr=learning_rate),
			  metrics=['accuracy'])
		return model
