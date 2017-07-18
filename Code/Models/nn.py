from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.optimizers import Adadelta, SGD


def modelA(nb_classes, learning_rate):
	model = Sequential()
	model.add(Flatten(input_shape=(3, 32, 32)))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='softmax'))
	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model


def modelA_weak(nb_classes, learning_rate):
	model = Sequential()
        model.add(Flatten(input_shape=(3, 32, 32)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='softmax'))
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model
