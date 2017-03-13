from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU


def conv_stack(filters, side, activation, model, input_shape=None):
	if not input_shape:
		model.add(Convolution2D(filters, side, side, border_mode='same'))
	else:
		model.add(Convolution2D(filters, side, side, border_mode='same', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(activation())


def cnn_cifar100(learning_rate):
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
	model.add(Dropout(0.3))
	model.add(MaxPooling2D())

	conv_stack(280, 1, ELU, model)
	conv_stack(300, 2, ELU, model)
	model.add(Dropout(0.4))
	model.add(MaxPooling2D())

	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(300))
	model.add(ELU())
	model.add(BatchNormalization())
	model.add(Dense(100))
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


if __name__ == "__main__":
	import keras
	keras.backend.set_image_dim_ordering('th')
	m = cnn_cifar100(1)
