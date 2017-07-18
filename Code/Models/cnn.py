from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.optimizers import Adadelta, SGD


def modelA(nb_classes, learning_rate):
	# As defined here: https://github.com/dribnet/kerosene/blob/master/examples/cifar100.py
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3, 32, 32)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model


def modelA_weak(nb_classes, learning_rate):
        model = Sequential()
        model.add(Convolution2D(8, 3, 3, border_mode='same',
                        input_shape=(3, 32, 32)))
        model.add(Activation('relu'))
        model.add(Convolution2D(8, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model


def modelB(img_rows=28, img_cols=28, nb_filters=64, nb_classes=10, learning_rate=1.0):
	model = Sequential()
	model.add(Convolution2D(nb_filters, 3, 3,
			border_mode='valid',
			input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def modelC(logits=False,input_ph=None, img_rows=28, img_cols=28, hidden_neurons=512, nb_classes=10):
	model = Sequential()
	model.add(Dense(hidden_neurons, input_shape=(img_rows * img_cols,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(hidden_neurons))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def model_atrous(img_rows=28, img_cols=28, nb_filters=64, nb_classes=10, learning_rate=1.0):
	model = Sequential()
	model.add(AtrousConvolution2D(nb_filters, 3, 3,
		border_mode='valid',
		input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(AtrousConvolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def model_separable(img_rows=28, img_cols=28, nb_filters=64, nb_classes=10, learning_rate=1.0):
	model = Sequential()
	model.add(SeparableConvolution2D(nb_filters, 3, 3,
		border_mode='valid',
		input_shape=(3, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(SeparableConvolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.compile(optimizer=Adadelta(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def modelSVHN(nb_classes, learning_rate):
        # As defined here: https://github.com/penny4860/SVHN-deep-digit-detector
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3, 32, 32)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model
