import common
import keras
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation
from keras.models import Model, load_model
from keras import backend as K
import data_load
from Models import densenet, resnet

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER', help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER', help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING', help='dataset. (default: cifar10)')
parser.add_argument('-t','--teacher', type=str, default="", metavar='STRING', help='path to teacher model')

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
	print("WEIGHT DECAY: {:.4f}".format(weight_decay))
	print("EPOCHS: {:3d}".format(epochs))
	print("DATASET: {:}".format(args.dataset))
	global num_classes
	dataObject = data_load.get_appropriate_data(args.dataset)(None, None)
	(xt, yt), (x_test, y_test) = dataObject.get_blackbox_data()
	x_train, _, x_val, _ = dataObject.validation_split(xt, yt, 0.2)
	print("== DONE! ==\n== BUILD MODEL... ==")
	is_mnist = (args.dataset == "mnist")
	student, cbks = densenet.densenet(n_classes=10, mnist=is_mnist, get_logits=False)
	print(student.summary())
	teacher_model = load_model(args.teacher)
	print("== GENERATING DATA FOR STUDENT MODEL... ==")
	y_train = teacher_model.predict(x_train, batch_size=1024)
	y_val = teacher_model.predict(x_val, batch_size=1024)
	print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
	datagen = dataObject.data_generator()
	datagen.fit(x_train)
	generator = datagen.flow(x_train, y_train, batch_size=batch_size)
	student.fit_generator(generator, steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, validation_data=(x_val, y_val))
	student.save('densenet_{:d}_{}.h5'.format(layers,args.dataset))
	print(student.evaluate(x_test, y_test))