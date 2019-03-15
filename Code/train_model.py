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

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER', help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER', help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING', help='dataset. (default: cifar10)')
parser.add_argument('-s','--smooth', type=float, default=0, help='Amount of label smoothening to be applied')
parser.add_argument('-g','--save_here', type=str, default="", metavar='STRING', help='path where trained model should be saved')
args = parser.parse_args()

if __name__ == '__main__':
	print("MODEL: Residual Network ({:2d} layers)".format(6 * args.stack_n + 2))
	print("DATASET: {:}".format(args.dataset))
	global num_classes
	dataObject = data_load.get_appropriate_data(args.dataset)(None, None)
	(x_train, y_train), (x_test, y_test) = dataObject.get_blackbox_data()
	(x_val, y_val) = dataObject.get_validation_data()
	is_mnist = (args.dataset == "mnist")
	if is_mnist:
		model, cbks = lenet.lenet_network(n_classes=10, is_mnist=is_mnist)
	else:
		model, cbks = resnet.residual_network(n_classes=10, stack_n=args.stack_n, mnist=is_mnist)
	print(model.summary())
	datagen = dataObject.data_generator(indeces=False)
	datagen.fit(x_train)
	if args.smooth > 0:
		y_train = y_train.clip(args.smooth / 9., 1. - args.smooth)
	generator = datagen.flow(x_train, y_train, batch_size=args.batch_size)
	model.fit_generator(generator, steps_per_epoch=50000 // args.batch_size + 1, epochs=args.epochs, callbacks=cbks, validation_data=(x_val, y_val))
	model.save(args.save_here)
	print(model.evaluate(x_test, y_test))