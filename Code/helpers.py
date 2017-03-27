import common

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


def jbda(X_train, Y_train, prefix, n_points, nb_classes = 100, pool_split=0.8):
	# Try loading cached copy, if available
	try:
		X_train_bm = np.load("__bm" + prefix + str(n_points) + "_x.npy")
		Y_train_bm = np.load("__bm" + prefix + str(n_points) + "_y.npy")
		X_train_pm = np.load("__pm" + prefix + str(n_points) + "_x.npy")
		Y_train_pm = np.load("__pm" + prefix + str(n_points) + "_y.npy")
		return X_train_bm, Y_train_bm, X_train_pm, Y_train_pm
	except:
		distr = {}
		for i in range(nb_classes):
			distr[i] = []
		if Y_train.shape[1] == nb_classes:
			for i in range(len(Y_train)):
				distr[np.argmax(Y_train[i])].append(i)
		else:
			for i in range(len(Y_train)):
				distr[Y_train[i][0]].append(i)
		X_train_bm_ret = []
		Y_train_bm_ret = []
		X_train_pm_ret = []
		Y_train_pm_ret = []
		for key in distr.keys():
			st = np.random.choice(distr[key], n_points, replace=False)
			bm = st[:int(len(st)*pool_split)]
			pm = st[int(len(st)*pool_split):]
			X_train_bm_ret.append(X_train[bm])
			Y_train_bm_ret.append(Y_train[bm])
			X_train_pm_ret.append(X_train[pm])
			Y_train_pm_ret.append(Y_train[pm])
		X_train_bm_ret = np.concatenate(X_train_bm_ret)
		Y_train_bm_ret = np.concatenate(Y_train_bm_ret)
		X_train_pm_ret = np.concatenate(X_train_pm_ret)
		Y_train_pm_ret = np.concatenate(Y_train_pm_ret)
		# Cache data for later use
		np.save("__bm" + prefix + str(n_points) + "_x.npy", X_train_bm_ret)
		np.save("__bm" + prefix + str(n_points) + "_y.npy", Y_train_bm_ret)
		np.save("__pm" + prefix + str(n_points) + "_x.npy", X_train_pm_ret)
		np.save("__pm" + prefix + str(n_points) + "_y.npy", Y_train_pm_ret)
		return X_train_bm_ret, Y_train_bm_ret, X_train_pm_ret, Y_train_pm_ret


def validation_split(X, y, validation_split=0.2):
	num_points = len(X)
	validation_indices = np.random.choice(num_points, int(num_points * validation_split))
	train_indices = list(set(range(num_points)) - set(validation_indices))
	X_train, y_train = X[train_indices], y[train_indices]
	X_val, y_val = X[validation_indices], y[validation_indices]
	return X_train, y_train, X_val, y_val
