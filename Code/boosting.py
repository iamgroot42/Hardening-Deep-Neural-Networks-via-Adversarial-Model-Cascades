import tqdm

import keras
from keras.models import *
from keras.layers import *

from keras.datasets import mnist
import numpy as np
import os
import helpers

import utils_cifar, utils_mnist, utils_svhn
from Models import cnn, sota

from keras.objectives import categorical_crossentropy
from tensorflow.python.platform import app
from keras.utils import np_utils

import tensorflow as tf
from keras import backend as K

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs')
flags.DEFINE_integer('num_iters', 2, 'Numer of iterations inside boosting algorithm')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('mode', 'train', '(train,test,finetune)')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_string('input_model_dir', './', 'path to input directory of models')
flags.DEFINE_string('output_model_dir', './', 'path to output directory of models')
flags.DEFINE_boolean('add_model', True, 'Add a model to the existing bag')
flags.DEFINE_string('data_x', './', 'path to numpy file of data for prediction')
flags.DEFINE_string('data_y', './', 'path to numpy file of labels for prediction')
flags.DEFINE_float('learning_rate', 0.001 ,'Learning rate for classifier')

#Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


class Boosting:
	def __init__(self, batch_size=16, nb_epochs=10, iters=2):
		self.models = []
		self.modelWeights = None
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.iters = iters

	def weightedCross(self, y_ph, y_pred, w_ph):
		return tf.multiply( w_ph , categorical_crossentropy(y_ph, y_pred))

	def fitWeighted(self, model, X, Y, W):
		model.fit(X, Y, batch_size=self.batch_size, epochs=self.nb_epochs, validation_split=0.2,sample_weight=W)

	def train(self, X, Y):
		modelWeights= np.zeros(len(self.models))
		dataPointWeights = np.ones(X.shape[0]) / X.shape[0]

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		with sess.as_default():
			for _ in range(self.iters):
				for i,m in enumerate(self.models):
					self.fitWeighted(m, X, Y, dataPointWeights)
					yp = m.predict(X)
					IPreds = (Y.argmax(axis=1) != yp.argmax(axis=1))
					epM = np.sum(dataPointWeights*IPreds)/np.sum(dataPointWeights)
					alpM = np.log((1-epM)/epM)
					dataPointWeights = dataPointWeights*np.exp(alpM*IPreds)
					modelWeights[i] = alpM
		self.modelWeights = modelWeights

	def predict(self, X):
		preds = [None for _ in range(len(self.models))]
		for i,m in enumerate(self.models):
			preds[i] = m.predict(X)
		Y = np.zeros_like(preds[0])
		weightT = 0.0
		for i,m in enumerate(self.models):
			Y += self.modelWeights[i]*preds[i]
			weightT += self.modelWeights[i]
		Y /= weightT
		return Y

	def load_models(self, data_dir, add_model):
		if add_model:
			self.models.append(add_model)
		for file in os.listdir(data_dir):
			if "npy" not in file:
				self.models.append(load_model(os.path.join(data_dir,file)))
		try:
			self.modelWeights = np.load(os.path.join(data_dir,"weights.npy"))
		except:
			print "Training first time"

	def save_models(self, data_dir):
		for i, model in enumerate(self.models):
			model.save(os.path.join(data_dir,str(i+1)))
		np.save(os.path.join(data_dir, "weights"), self.modelWeights)

	def accuracy(self, Y, Y_):
		num = (np.argmax(Y,axis=1) == np.argmax(Y_,axis=1)).sum()
		den = len(Y)
		print num, den
		return num/den


def main(argv=None):
	# Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'th':
                keras.backend.set_image_dim_ordering('th')

	boost = Boosting(FLAGS.batch_size, FLAGS.nb_epochs, FLAGS.num_iters)
	add_model = None
	if FLAGS.add_model:
		if FLAGS.dataset == 'cifar100':
			add_model = sota.cifar_svhn(FLAGS.learning_rate, 100)
		elif FLAGS.dataset == 'svhn':
			add_model = sota.cifar_svhn(FLAGS.learning_rate, 10)
		elif FLAGS.dataset == 'mnist':
			add_model = sota.mnist(FLAGS.learning_rate, 10)
		else:
			print "Invalid dataset; exiting"
			exit()
	boost.load_models(FLAGS.input_model_dir, add_model)
	#Training mode
	if FLAGS.mode in ['train', 'finetune']:
		# Load data
		if FLAGS.dataset == 'cifar100':
			X, Y, _, _ = utils_cifar.data_cifar()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, 100)
		elif FLAGS.dataset == 'mnist':
			X, Y, _, _ = utils_mnist.data_mnist()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 5000, 10)
		elif FLAGS.dataset == 'svhn':
			X, Y, _, _ = utils_svhn.data_svhn()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 4000, 10)
		else:
			print "Invalid dataset; exiting"
			exit()
		#Finetune mode
		if FLAGS.mode == 'finetune':
			X_train_p = np.concatenate((X_train_p, np.load(FLAGS.data_x)))
			Y_train_p = np.concatenate((Y_train_p, np.load(FLAGS.data_y)))
		boost.train(X_train_p, Y_train_p)
		# Print validation accuracy
		X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
		predicted = np.argmax(boost.predict(X_val),1)
		true = np.argmax(y_val,1)
		acc = (100*(predicted==true).sum()) / float(len(y_val))
		print "Final validation accuracy", acc
		# Save models
		boost.save_models(FLAGS.output_model_dir)
	#Testing mode
	elif FLAGS.mode == 'test':
		X = np.load(FLAGS.data_x)
		Y = np.load(FLAGS.data_y)
		boost.load_models(FLAGS.input_model_dir)
		Y_ = boost.predict(X)
		print "Misclassification accuracy",(100-boost.accuracy(Y, Y_))
	else:
		print "Invalid option"


if __name__ == "__main__":
	app.run()
