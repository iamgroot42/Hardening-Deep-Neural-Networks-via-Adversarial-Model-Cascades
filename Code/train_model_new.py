import common

import keras
import argparse
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K

import data_load
from Models import resnet, lenet, densenet

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
				help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
				help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
				help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
				help='dataset. (default: cifar10)')
parser.add_argument('-s','--smooth', type=float, default=0,
				help='Amount of label smoothening to be applied')

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4


if __name__ == '__main__':

	print("========================================")
	print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+2))
	print("BATCH SIZE: {:3d}".format(batch_size))
	print("WEIGHT DECAY: {:.4f}".format(weight_decay))
	print("EPOCHS: {:3d}".format(epochs))
	print("DATASET: {:}".format(args.dataset))

	print("== LOADING DATA... ==")
	# load data
	global num_classes

	dataObject = data_load.get_appropriate_data(args.dataset)(None, None)
	(xt, yt), (x_test, y_test) = dataObject.get_blackbox_data()
	x_train, y_train, x_val, y_val = dataObject.validation_split(xt, yt, 0.2)

	print("== DONE! ==\n== BUILD MODEL... ==")
	is_mnist = (args.dataset == "mnist")
	# build network

	# RESNET:
	#model, cbks = resnet.residual_network(n_classes=10, stack_n=stack_n, mnist=is_mnist)

	# LENET:
	#model, cbks = lenet.lenet_network(n_classes=10)

	# DENSENET
	model, cbks = densenet.densenet(n_classes=10, mnist=is_mnist)

	# print model architecture if you need.
	print(model.summary())

	# set data augmentation
	print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
	datagen = dataObject.data_generator()
	datagen.fit(x_train)

	# add label-noise as specifed
	if args.smooth > 0:
		y_train = y_train.clip(args.smooth / 9., 1. - args.smooth)

	# start training
	generator = datagen.flow(x_train, y_train, batch_size=batch_size)
	model.fit_generator(generator, steps_per_epoch=iterations,
						 epochs=epochs,
						 callbacks=cbks,
						 validation_data=(x_val, y_val))

	model.save('densenet_{:d}_{}.h5'.format(layers,args.dataset))
	print(model.evaluate(x_test, y_test))