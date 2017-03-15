from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.optimizers import Adadelta, SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.initializers import he_normal


def conv_stack(filters, side, activation, model, input_shape=None):
	if not input_shape:
		model.add(Convolution2D(filters, side, side, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	else:
		model.add(Convolution2D(filters, side, side, border_mode='same', input_shape=input_shape, W_regularizer=l2(0.001), init=he_normal()))
	model.add(BatchNormalization())
	model.add(activation())


def cnn_cifar100(learning_rate):
	model = Sequential()
	conv_stack(192, 5, ELU, model,(3, 32, 32))
	model.add(MaxPooling2D())

	conv_stack(192, 1, ELU, model)
	conv_stack(240, 3, ELU, model)
	model.add(Dropout(0.3))
	model.add(MaxPooling2D())

	conv_stack(240, 1, ELU, model)
	conv_stack(260, 2, ELU, model)
	model.add(Dropout(0.4))
	model.add(MaxPooling2D())

	conv_stack(260, 1, ELU, model)
	conv_stack(280, 2, ELU, model)
	model.add(Dropout(0.5))
	model.add(MaxPooling2D())

	conv_stack(280, 1, ELU, model)
	conv_stack(300, 2, ELU, model)
	model.add(Dropout(0.6))
	model.add(MaxPooling2D())

	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(300, W_regularizer=l2(0.001), init=he_normal()))
	model.add(ELU())
	model.add(BatchNormalization())
	model.add(Dense(100, W_regularizer=l2(0.001), init=he_normal()))
	model.add(Activation('softmax'))
	model.compile(optimizer=SGD(lr=learning_rate,momentum=0.9),loss='categorical_crossentropy', metrics=['accuracy'])
	#model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def allcnn_cifar100(learning_rate):
	# All-CNN, as described in : https://arxiv.org/pdf/1412.6806.pdf
	model = Sequential()
	model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(3,32,32), W_regularizer=l2(0.005), init=he_normal()))
	model.add(Activation('relu'))
	model.add(Convolution2D(96, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(Convolution2D(96, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal(), subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal(), subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(192, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	model.add(Convolution2D(192, 1, 1, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	model.add(Convolution2D(10, 3, 3, border_mode='same', W_regularizer=l2(0.001), init=he_normal()))
	model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax'))
	model.compile(optimizer=SGD(lr=learning_rate,momentum=0.9),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


if __name__ == "__main__":
	import keras
	keras.backend.set_image_dim_ordering('th')
	m = cnn_cifar100(1)
