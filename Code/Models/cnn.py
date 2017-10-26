from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AtrousConvolution2D, SeparableConvolution2D, AveragePooling2D
from keras.optimizers import Adadelta, SGD


def proxy(shape, nb_classes, distill=False):
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='same',
			input_shape=(shape[0], shape[1], shape[2])))
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
	
	if distill:
		def fn(correct, predicted):
			return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)
		
		model.compile(loss=fn,
			optimizer=sgd,
			metrics=['accuracy'])
	else:
		model.add(Activation('softmax'))
		model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=Adadelta(lr=learning_rate),
			metrics=['accuracy'])
	return model
