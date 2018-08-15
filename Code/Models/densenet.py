import keras
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from keras import regularizers


def scheduler(epoch):
	if epoch < 150:
		return 0.1
	if epoch < 225:
		return 0.01
	return 0.001

def densenet(n_classes=10, mnist=False, get_logits=False):
	growth_rate        = 12
	depth              = 100
	compression        = 0.5
	weight_decay       = 1e-4
	def densenet_mdl(img_input, classes_num):
		def bn_relu(x):
			x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
			x = Activation('relu')(x)
			return x

		def bottleneck(x):
			channels = growth_rate * 4
			x = bn_relu(x)
			x = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
			x = bn_relu(x)
			x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
			return x

		def single(x):
			x = bn_relu(x)
			x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
			return x

		def transition(x, inchannels):
			outchannels = int(inchannels * compression)
			x = bn_relu(x)
			x = Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
			x = AveragePooling2D((2,2), strides=(2, 2))(x)
			return x, outchannels

		def dense_block(x,blocks,nchannels):
			concat = x
			for i in range(blocks):
				x = bottleneck(concat)
				concat = concatenate([x,concat], axis=-1)
				nchannels += growth_rate
			return concat, nchannels

		def dense_layer(x):
			return Dense(classes_num,kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)

		nblocks = (depth - 4) // 6
		nchannels = growth_rate * 2

		x = Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(img_input)

		x, nchannels = dense_block(x,nblocks,nchannels)
		x, nchannels = transition(x,nchannels)
		x, nchannels = dense_block(x,nblocks,nchannels)
		x, nchannels = transition(x,nchannels)
		x, nchannels = dense_block(x,nblocks,nchannels)
		x, nchannels = transition(x,nchannels)
		x = bn_relu(x)
		x = GlobalAveragePooling2D()(x)
		x = dense_layer(x)
		return x

	cbks = [LearningRateScheduler(scheduler),
                ModelCheckpoint('./checkpoint-densenet-{epoch}.h5', save_best_only=False, mode='auto', period=10)]
	img_input = Input(shape=(32, 32, 3))
	if mnist == True:
		img_input = Input(shape=(28, 28, 1))

	logits = densenet_mdl(img_input, n_classes)
	model = None
	if get_logits:
		densenet = Model(img_input, logits)
		return densenet, cbks
	else:
		output = Activation('softmax')(logits)
		densenet = Model(img_input, output)

	sgd = optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True)
	densenet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return densenet, cbks
