import common

import tensorflow as tf
import numpy as np
import copy

# Set seed for reproducability
np.random.seed(42)

def tf_model_loss(y, model):
	op = model.op
	if "softmax" in str(op).lower():
		logits, = op.inputs
	else:
		logits = model

	out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
	return out


def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
	y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
	y = y / tf.reduce_sum(y, 1, keep_dims=True)
	loss = tf_model_loss(y, predictions)
	grad, = tf.gradients(loss, x)
	adv_x = tf.stop_gradient(x + (eps * tf.sign(grad)))
	if (clip_min is not None) and (clip_max is not None):
		adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
	return adv_x


def jbda(X_train, Y_train, prefix, n_points, nb_classes = 100, pool_split=0.8):
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
	return X_train_bm_ret, Y_train_bm_ret, X_train_pm_ret, Y_train_pm_ret


def validation_split(X, y, validation_split=0.2):
	num_points = len(X)
	validation_indices = np.random.choice(num_points, int(num_points * validation_split))
	train_indices = list(set(range(num_points)) - set(validation_indices))
	X_train, y_train = X[train_indices], y[train_indices]
	X_val, y_val = X[validation_indices], y[validation_indices]
	return X_train, y_train, X_val, y_val


def model_argmax(sess, x, predictions, samples):
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def apply_perturbations(i, j, X, increase, theta, clip_min, clip_max):
	if increase:
		X[0, i] = np.minimum(clip_max, X[0, i] + theta)
		X[0, j] = np.minimum(clip_max, X[0, j] + theta)
	else:
		X[0, i] = np.maximum(clip_min, X[0, i] - theta)
		X[0, j] = np.maximum(clip_min, X[0, j] - theta)
	return X


def saliency_map(grads_target, grads_other, search_domain, increase):
	nf = len(grads_target)
	invalid = list(set(range(nf)) - search_domain)
	grads_target[invalid] = 0
	grads_other[invalid] = 0
	target_sum = grads_target.reshape((1, nf)) + grads_target.reshape((nf, 1))
	other_sum = grads_other.reshape((1, nf)) + grads_other.reshape((nf, 1))
	if increase:
		scores_mask = ((target_sum > 0) & (other_sum < 0))
	else:
		scores_mask = ((target_sum < 0) & (other_sum > 0))
	scores = scores_mask * (-target_sum * other_sum)
	np.fill_diagonal(scores, 0)
	best = np.argmax(scores)
	p1, p2 = best % nf, best // nf
	search_domain.remove(p1)
	search_domain.remove(p2)
	return p1, p2, search_domain


def jacobian(sess, x, grads, target, X, nb_features, nb_classes):
	feed_dict = {x: X}
	jacobian_val = np.zeros((nb_classes, nb_features), dtype=np.float32)
	for class_ind, grad in enumerate(grads):
		run_grad = sess.run(grad, feed_dict)
		jacobian_val[class_ind] = np.reshape(run_grad, (1, nb_features))
	other_classes = utils.other_classes(nb_classes, target)
	grad_others = np.sum(jacobian_val[other_classes, :], axis=0)
	return jacobian_val[target], grad_others


def jsma(sess, x, predictions, grads, sample, target, theta, gamma, clip_min, clip_max):
	adv_x = copy.copy(sample)
	nb_features = np.product(adv_x.shape[1:])
	original_shape = adv_x.shape
	adv_x = np.reshape(adv_x, (1, nb_features))
	max_iters = np.floor(nb_features * gamma / 2)
	increase = bool(theta > 0)
	if increase:
		search_domain = set([i for i in xrange(nb_features)
							 if adv_x[0, i] < clip_max])
	else:
		search_domain = set([i for i in xrange(nb_features)
							 if adv_x[0, i] > clip_min])
	iteration = 0
	adv_x_original_shape = np.reshape(adv_x, original_shape)
	current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape)
	while (current != target and iteration < max_iters and
		   len(search_domain) > 1):
		adv_x_original_shape = np.reshape(adv_x, original_shape)
		grads_target, grads_others = jacobian(sess, x, grads, target, adv_x_original_shape,b_features, FLAGS.nb_classes)
		i, j, search_domain = saliency_map(grads_target, grads_others, search_domain, increase)
		adv_x = apply_perturbations(i, j, adv_x, increase, theta, clip_min, clip_max)
		current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape)
		iteration = iteration + 1
	percent_perturbed = float(iteration * 2) / nb_features
	if current == target:
		return np.reshape(adv_x, original_shape), 1, percent_perturbed
	else:
		return np.reshape(adv_x, original_shape), 0, percent_perturbed

def random_targets(gt, nb_classes):
    if len(gt.shape) > 1:
        gt = np.argmax(gt, axis=1)

    result = np.zeros(gt.shape)

    for class_ind in xrange(nb_classes):
        in_cl = gt == class_ind
        result[in_cl] = np.random.choice(other_classes(nb_classes, class_ind))

    return np_utils.to_categorical(np.asarray(result), nb_classes)

def jsma_batch(sess, x, pred, grads, X, theta, gamma, clip_min, clip_max, nb_classes, targets=None):
	X_adv = np.zeros(X.shape)
	for ind, val in enumerate(X):
		val = np.expand_dims(val, axis=0)
		if targets is None:
			gt = model_argmax(sess, x, pred, val)
			target = random_targets(gt, nb_classes)[0]
		else:
			target = targets[ind]

		X_adv[ind], _, _ = jsma(sess, x, pred, grads, val, np.argmax(target), theta, gamma, clip_min, clip_max)
	return np.asarray(X_adv, dtype=np.float32)

def other_classes(nb_classes, class_ind):
    other_classes_list = list(range(nb_classes))
    other_classes_list.remove(class_ind)
    return other_classes_list
