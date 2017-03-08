import common

# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

import keras

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Reshape
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

import numpy as np
import math

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', 'Path where model is to be saved')
flags.DEFINE_integer('batch_size', 512 , 'Batch size')
flags.DEFINE_boolean('nice', False , 'Nice')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for training')


def generator_model():
	model = Sequential()
	model.add(Dense(input_dim=100, output_dim=1024))
	model.add(Activation('tanh'))
	model.add(Dense(128*8*8))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Reshape((128, 8, 8), input_shape=(128*8*8,)))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(64, 5, 5, border_mode='same'))
	model.add(Activation('tanh'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(3, 5, 5, border_mode='same'))
	model.add(Activation('tanh'))
	return model


def discriminator_model():
	model = Sequential()
	model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(3, 32, 32)))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(128, 5, 5))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('tanh'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model


def generator_containing_discriminator(generator, discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	return model


def combine_images(generated_images):
	num = generated_images.shape[0]
	width = int(math.sqrt(num))
	height = int(math.ceil(float(num)/width))
	shape = generated_images.shape[2:]
	image = np.zeros((height*shape[0], width*shape[1]),
					 dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		i = int(index/width)
		j = index % width
		image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
			img[0, :, :]
	return image


def train(X_train, y_train, X_test, y_test, BATCH_SIZE):
	X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
	discriminator = discriminator_model()
	generator = generator_model()
	discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

	d_optim = SGD(lr=FLAGS.learning_rate, momentum=0.9, nesterov=True)
	g_optim = SGD(lr=FLAGS.learning_rate, momentum=0.9, nesterov=True)

	generator.compile(loss='binary_crossentropy', optimizer="SGD")
	discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)

	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
	noise = np.zeros((BATCH_SIZE, 100))

	for epoch in range(100):
		print("Epoch is", epoch)
		print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))

		for index in range(int(X_train.shape[0]/BATCH_SIZE)):

			for i in range(BATCH_SIZE):
				noise[i, :] = np.random.uniform(-1, 1, 100)

			image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
			
			generated_images = generator.predict(noise, verbose=0)
			print "\n\n\n\n"
			print image_batch.shape
			print generated_images.shape
			print "\n\n\n"
			# print(image_batch.shape)
			# print(generated_images.shape)
			X = np.concatenate((image_batch, generated_images))
			y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

			d_loss = discriminator.train_on_batch(X, y)

			print("batch %d d_loss : %f" % (index, d_loss))

			for i in range(BATCH_SIZE):
				noise[i, :] = np.random.uniform(-1, 1, 100)

			discriminator.trainable = False
			g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)
			discriminator.trainable = True

			print("batch %d g_loss : %f" % (index, g_loss))

			if index % 10 == 9:
				generator.save_weights('generator', True)
				discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
	generator = generator_model()
	generator.compile(loss='binary_crossentropy', optimizer="SGD")
	generator.load_weights('generator')

	if nice:
		discriminator = discriminator_model()
		discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
		discriminator.load_weights('discriminator')
		noise = np.zeros((BATCH_SIZE*20, 100))
		for i in range(BATCH_SIZE*20):
			noise[i, :] = np.random.uniform(-1, 1, 100)
		generated_images = generator.predict(noise, verbose=1)
		d_pret = discriminator.predict(generated_images, verbose=1)
		index = np.arange(0, BATCH_SIZE*20)
		index.resize((BATCH_SIZE*20, 1))
		pre_with_index = list(np.append(d_pret, index, axis=1))
		pre_with_index.sort(key=lambda x: x[0], reverse=True)
	else:
		noise = np.zeros((BATCH_SIZE, 100))
		for i in range(BATCH_SIZE):
			noise[i, :] = np.random.uniform(-1, 1, 100)
		generated_images = generator.predict(noise, verbose=1)

	return generated_images


if __name__ == "__main__":

	import utils_cifar

	xtr, ytr, xte, yte = utils_cifar.data_cifar()
	print xtr.shape
	
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	if FLAGS.mode == "train":
		train(xtr, yte, xte, yte, FLAGS.batch_size)
	elif FLAGS.mode == "generate":
		generate(FLAGS.batch_size, nice=args.nice)
