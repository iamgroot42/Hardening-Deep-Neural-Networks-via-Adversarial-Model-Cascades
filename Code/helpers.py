from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np


def tf_model_loss(y, model):
	op = model.op
	if "softmax" in str(op).lower():
		logits, = op.inputs
	else:
		logits = model

	out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
	return out


def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
	# Compute loss
	y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
	y = y / tf.reduce_sum(y, 1, keep_dims=True)
	loss = tf_model_loss(y, predictions)
	# Define gradient of loss wrt input
	grad, = tf.gradients(loss, x)
	# Add perturbation to original example to obtain adversarial example
	adv_x = tf.stop_gradient(x + (eps * tf.sign(grad)))
	# If clipping is needed, reset all values outside of [clip_min, clip_max]
	if (clip_min is not None) and (clip_max is not None):
		adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
	return adv_x


def jbda(X_train, Y_train, prefix, n_points=200):
	# Try loading cached copy, if available
	try:
		X_train = np.load("__" + prefix + str(n_points) + "_x.npy")
		Y_train = np.load("__" + prefix + str(n_points) + "_y.npy")
		return X_train, Y_train
	except:
		n_classes = 10
		distr = {}
		for i in range(n_classes):
			distr[i] = []
		if Y_train.shape[1] == n_classes:
			for i in range(len(Y_train)):
				distr[np.argmax(Y_train[i])].append(i)
		else:
			for i in range(len(Y_train)):
				distr[Y_train[i][0]].append(i)
		X_train_ret = []
		Y_train_ret = []
		for key in distr.keys():
			for i in distr[key]:
				X_train_ret.append(X_train[i])
				Y_train_ret.append(Y_train[i])
		X_train_ret = np.array(X_train_ret)
		Y_train_ret = np.array(Y_train_ret)
		# Cache data for later use
		np.save("__" + prefix + str(n_points) + "_x.npy", X_train_ret)
		np.save("__" + prefix + str(n_points) + "_y.npy", Y_train_ret)
		return X_train_ret, Y_train_ret
