import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K


def scheduler(epoch):
	if epoch < 81:
		return 0.1
	if epoch < 122:
		return 0.01
	return 0.001


def residual_network(n_classes=10, stack_n=5, mnist=False, get_logits=False):
	weight_decay       = 1e-4
	img_input = Input(shape=(3, 32, 32))
	if mnist == True:
		img_input = Input(shape=(1, 28, 28))

	def residual_block(x, o_filters, increase=False):
		stride = (1,1)
		if increase:
			stride = (2,2)

		o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
		conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
						kernel_initializer="he_normal",
						kernel_regularizer=regularizers.l2(weight_decay))(o1)
		o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
		conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
						kernel_initializer="he_normal",
						kernel_regularizer=regularizers.l2(weight_decay))(o2)
		if increase:
			projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
						kernel_initializer="he_normal",
						kernel_regularizer=regularizers.l2(weight_decay))(o1)
			block = add([conv_2, projection])
		else:
			block = add([conv_2, x])
		return block

	# build model ( total layers = stack_n * 3 * 2 + 2 )
	# stack_n = 5 by default, total layers = 32
	# input: 32x32x3 output: 32x32x16
	x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
			   kernel_initializer="he_normal",
			   kernel_regularizer=regularizers.l2(weight_decay))(img_input)

	# input: 32x32x16 output: 32x32x16
	for _ in range(stack_n):
		x = residual_block(x, 16, False)

	# input: 32x32x16 output: 16x16x32
	x = residual_block(x, 32, True)
	for _ in range(1, stack_n):
		x = residual_block(x, 32, False)

	# input: 16x16x32 output: 8x8x64
	x = residual_block(x, 64, True)
	for _ in range(1, stack_n):
		x = residual_block(x, 64, False)

	x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
	x = Activation('relu')(x)
	x = GlobalAveragePooling2D()(x)

	# input: 64 output: 10
	logits = Dense(n_classes,activation='softmax',kernel_initializer="he_normal",
			  kernel_regularizer=regularizers.l2(weight_decay))(x)

	output = Activation('softmax')(logits)
	cbks = [LearningRateScheduler(scheduler),
                ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)]
	resnet = None
	if get_logits:
		resnet = Model(img_input, logits)
		return resnet, cbks
	else:
		resnet = Model(img_input, output)
	sgd = optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True)
	resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return resnet, cbks


if __name__ == "__main__":
	model = residual_network()
