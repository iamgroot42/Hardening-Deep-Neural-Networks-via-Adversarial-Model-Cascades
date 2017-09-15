#Author : Divam Gupta

import tqdm

import keras
from keras.models import *
from keras.layers import *

from keras.datasets import mnist
import numpy as np
import os
import helpers
import utils_cifar

from keras.objectives import categorical_crossentropy
from keras.utils import np_utils

import tensorflow as tf
from keras import backend as K

sess = tf.Session()
K.set_session(sess)


class Boosting:
	def __init__(self, batch_size=32, nb_epochs=10, optimizer_lr=0.5):
		self.models = []
		self.modelWeights = None
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.optimizer_lr = optimizer_lr

	def weightedCross(self, y_ph, y_pred, w_ph):
		return tf.multiply( w_ph , categorical_crossentropy(y_ph, y_pred))

	def fitWeighted(self, model, X, Y, W):
		model.fit(X, Y, sample_weight=W, epochs=self.nb_epochs, batch_size=self.batch_size)
		return
		x_ph = tf.placeholder(tf.float32, shape=[self.batch_size] + list(X.shape[1:]))
		y_ph = tf.placeholder(tf.float32, shape=[self.batch_size] + list(Y.shape[1:]))
		w_ph = tf.placeholder(tf.float32, shape=(self.batch_size, ))

		y_pred = model(x_ph)
		loss = tf.reduce_mean(self.weightedCross(y_ph, y_pred, w_ph))
		train_step = tf.train.GradientDescentOptimizer(self.optimizer_lr).minimize(loss)
		for _ in tqdm.tqdm(range(self.nb_epochs)):
			for i in range(0, X.shape[0]-self.batch_size, self.batch_size):
				yB = Y[i:i+self.batch_size]
				xB = X[i:i+self.batch_size]
				wB = W[i:i+self.batch_size]
				train_step.run(feed_dict={ y_ph:yB , x_ph:xB , w_ph: wB})

	def train(self, X, Y):
		modelWeights= np.zeros(len(self.models))
		dataPointWeights = np.ones(X.shape[0]) / X.shape[0]

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		with sess.as_default():
			for it in range(10):
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
		preds = [None for _ in len(self.models)]
		for i,m in enumerate(self.models):
			preds[i] = m.predict(X)
		Y = np.zeros_like(preds[0])
		weightT = 0.0
		for i,m in enumerate(self.models):
			Y += self.modelWeights[i]*preds[i]
			weightT += self.modelWeights[i]
		Y /= weightT
		return Y

	def load_models(self, data_dir):
		for file in os.listdir(data_dir):
			self.models.append(load_model(data_dir + file))


if __name__ == "__main__":
	import sys
	n_classes = 100
	boost = Boosting(int(sys.argv[2]),int(sys.argv[3]), float(sys.argv[4]))
	boost.load_models(sys.argv[1])
	X, Y, _, _ = utils_cifar.data_cifar()
        x_train, y_train, x_test,  y_test = helpers.jbda(X, Y, "train", 500, n_classes)
	boost.train(x_train, y_train)
	boost.predict(x_test)

