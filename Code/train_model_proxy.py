import common

import keras
import argparse
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation
from keras.models import Model, load_model
from keras.utils import np_utils
from keras import backend as K

from cleverhans.utils_keras import KerasModelWrapper

import data_load, helpers
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
parser.add_argument('-s','--save_here', type=str, default="", metavar='STRING',
				 help='path where trained model should be saved')
parser.add_argument('-m','--mode', type=str, default="train", metavar='STRING',
				help='train/finetune model')
parser.add_argument('-l','--learning_rate', type=float, default=1e-1, metavar='FLOAT',
				help='learning rate')
parser.add_argument('-a','--attack', type=str, default="", metavar='STRING',
				help='attacks to be used while adversarial training')
parser.add_argument('-z','--label_smooth', type=float, default=0.0, metavar='FLOAT',
				help='amount of label smoothening to be applied')

args = parser.parse_args()


if __name__ == '__main__':

	batch_size         = args.batch_size
	epochs             = args.epochs
	assert(len(args.save_here) > 0, "Provide a path to save model")

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
	x_train, y_train, x_val, y_val = dataObject.validation_split(x_data, y_data, 0.1)
	iterations         = len(x_train) // batch_size + 1
	if args.label_smooth:
		y_train = y_train.clip(args.label_smooth / 9., 1. - args.label_smooth)

	print("== DONE! ==\n== BUILD MODEL... ==")

	# build proxy model (or load if fine-tuning)
	is_mnist = (args.dataset == "mnist")
	proxy = cnn.proxy(n_classes=10, mnist=is_mnist, learning_rate=args.learning_rate)
	if args.mode == "finetune":
		proxy = load_model(args.save_here)
		K.set_value(proxy.optimizer.lr, args.learning_rate)

	# set data augmentation
	print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
	datagen = dataObject.data_generator()
	datagen.fit(x_train)

	# start training proxy model
	attacks = args.attack.split(',')
	if len(attacks) > 1:
		attacks = attacks[1:]
		attack_params = []
		clever_wrapper = KerasModelWrapper(proxy)
		for attack in attacks:
			attack_params.append(helpers.get_appropriate_attack(args.dataset, dataObject.get_range(), attack,
				clever_wrapper, common.sess, harden=True, attack_type="black"))
	else:
		attack_params=None
	helpers.customTrainModel(proxy, x_train, y_train, x_val, y_val, datagen, epochs, None, batch_size, attack_params)

	print("== SAVING AND EVALUATING MODEL ==")
	proxy.save(args.save_here)
	print(proxy.evaluate(x_test, y_test))

