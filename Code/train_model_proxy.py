import common

import keras
import argparse
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation
from keras.models import Model, load_model
from keras.utils import np_utils
from keras import backend as K

import data_load
from Models import densenet, resnet, cnn

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
				help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
				help='epochs(default: 200)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
				help='dataset. (default: cifar10)')
parser.add_argument('-t','--blackbox', type=str, default="", metavar='STRING',
				help='path to blackbox model')
parser.add_argument('-k','--distill', type=bool, default=False, metavar='BOOLEAN',
				help='use distillation (probabilities) while training proxy?')

args = parser.parse_args()


if __name__ == '__main__':

	batch_size         = args.batch_size
	epochs             = args.epochs

	print("========================================")
	print("MODEL: Proxy Model")
	print("BATCH SIZE: {:3d}".format(batch_size))
	print("EPOCHS: {:3d}".format(epochs))
	print("DATASET: {:}".format(args.dataset))

	print("== LOADING DATA... ==")
	# load data
	global num_classes

	# Load blackbox model
	api_model = load_model(args.blackbox)

	# Get predictions from teacher_model
	print("== GENERATING DATA FOR PROXY MODEL... ==")
	x_data = data_load.get_proxy_data(args.dataset)
	y_data = api_model.predict(x_data, batch_size=1024)

	# convert to 1-hot if proxy is assumed to not have access to per-class probabilities
	convert_to_onehot = lambda vector: np_utils.to_categorical(np.argmax(vector, axis=1), 10)
	if not args.distill:
		print("NOT USING PROBABILITIES RETURNED BY BLACKBOX MODEL")
		y_data = convert_to_onehot(y_data)

	dataObject = data_load.get_appropriate_data(args.dataset)(None, None)
	_, (x_test, y_test) = dataObject.get_blackbox_data()
	x_train, y_train, x_val, y_val = dataObject.validation_split(x_data, y_data, 0.2)
	iterations         = len(x_train) // batch_size + 1

	print("== DONE! ==\n== BUILD MODEL... ==")

	# build proxy model
	is_mnist = (args.dataset == "mnist")
	proxy = cnn.proxy(n_classes=10, mnist=is_mnist, learning_rate=1e-1)

	# print model architecture if you need.
	print(proxy.summary())

	# set data augmentation
	print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
	datagen = dataObject.data_generator()
	datagen.fit(x_train)

	# start training proxy model
	generator = datagen.flow(x_train, y_train, batch_size=batch_size)
	proxy.fit_generator(generator, steps_per_epoch=iterations,
						 epochs=epochs,
						 validation_data=(x_val, y_val))

	proxy.save('proxy_{}.h5'.format(args.dataset))
	print(proxy.evaluate(x_test, y_test))

