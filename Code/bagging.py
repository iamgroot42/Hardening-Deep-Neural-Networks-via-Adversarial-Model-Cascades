import numpy as np

import keras
from keras.models import load_model

from keras.objectives import categorical_crossentropy
from keras.utils import np_utils

import utils_cifar, utils_mnist, utils_svhn
import helpers
import os

import tensorflow as tf
from tensorflow.python.platform import flags

#Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs')
flags.DEFINE_integer('sample_ratio', 0.5, 'Percentage of sample to be taken per model for training')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('mode', 'train', '(train,test,finetune)')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_string('model_dir', './', 'path to directory of models')
flags.DEFINE_string('data_x', './', 'path to numpy file of data for prediction')
flags.DEFINE_string('data_y', './', 'path to numpy file of labels for prediction')

class Bagging:
	def __init__(self, n_classes, sample_ratio, batch_size, nb_epochs):
		self.n_classes = n_classes
		self.sample_ratio = sample_ratio
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.models = []

	def load_models(self, data_dir):
		self.models = []
		for file in os.listdir(data_dir):
			self.models.append(load_model(os.path.join(data_dir,file)))

	def train(self, X, Y, data_dir):
		subsets = []
		for i in range(len(self.models)):
			subsets.append(np.random.choice(len(Y), int(len(Y) * self.sample_ratio)))
		for i, subset in enumerate(subsets):
			x_sub = X[subset]
			y_sub = Y[subset]
			datagen = utils_cifar.augmented_data(x_sub)
			X_tr, y_tr, X_val, y_val = helpers.validation_split(x_sub, y_sub, 0.2)
			self.models[i].fit_generator(datagen.flow(X_tr, y_tr,
							batch_size=self.batch_size),
							steps_per_epoch=X_tr.shape[0] // self.batch_size,
							epochs=self.nb_epochs,
							validation_data=(X_val, y_val))
			accuracy = self.models[i].evaluate(X_val, y_val, batch_size=self.batch_size)
			print("\nValidation accuracy for bag" + str(i) + " model: " + str(accuracy[1]*100))
			self.models[i].save(data_dir + "bag" + str(i))

	def predict(self, predict_on):
		predictions = []
		for model in self.models:
			predictions.append(model.predict(predict_on))
		ultimate = [ {i:0 for i in range(self.n_classes)} for j in range(len(predict_on))]
		for prediction in predictions:
			for i in range(len(prediction)):
				ultimate[i][np.argmax(prediction[i])] += 1
		predicted = []
		for u in ultimate:
			voted = sorted(u, key=u.get, reverse=True)
			predicted.append(voted[0])
		predicted = keras.utils.to_categorical(np.array(predicted), self.n_classes)
		return predicted


if __name__ == "__main__":
	bag = None
	if FLAGS.dataset == 'cifar100':
		bag = Bagging(100, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)
	elif FLAGS.dataset == 'mnist':
		bag = Bagging(10, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)
	elif FLAGS.dataset == 'svhn':
		bag = Bagging(10, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)
	else:
		print "Invalid dataset specified. Exiting"
		exit()
	#Training mode
	if FLAGS.mode in ['train', 'finetune']:
		X_train_p,Y_train_p = None, None
		# Load data
		if FLAGS.dataset == 'cifar100':
			X, Y, _, _ = utils_cifar.data_cifar()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, 100)
		elif FLAGS.dataset == 'mnist':
			X, Y, _, _ = utils_mnist.data_mnist()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 5000, 10)
		else:
			X, Y, _, _ = utils_svhn.data_svhn()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 4000, 10)
		#Finetune mode
		if FLAGS.mode == 'finetune':
			X_train_p = np.concatenate((X_train_p, np.load(FLAGS.data_x)))
			Y_train_p = np.concatenate((Y_train_p, np.load(FLAGS.data_y)))
		# Load placeholder models
		bag.load_models(FLAGS.model_dir)
		# Train data
		bag.train(X_train_p, Y_train_p, FLAGS.model_dir, True)
		# Print validation accuracy
		X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
		predicted = np.argmax(bag.predict(X_val),1)
		true = np.argmax(y_val,1)
		acc = (100*(predicted==true).sum()) / float(len(y_val))
		print "Final validation accuracy", acc
	#Testing mode
	elif FLAGS.mode == 'test':
		X = np.load(FLAGS.data_x)
		Y = np.load(FLAGS.data_y)
		bag.load_models(FLAGS.model_dir)
		predicted = np.argmax(bag.predict(X),1)
		Y = np.argmax(Y, 1)
		acc = (100*(predicted==Y).sum()) / float(len(Y))
		print "Misclassification accuracy",(100-acc)
	else:
		print "Invalid option"
