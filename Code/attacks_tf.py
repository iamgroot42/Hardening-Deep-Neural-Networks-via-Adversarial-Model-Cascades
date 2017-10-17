from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings

import utils_tf
import helpers
import utils

def apply_perturbations(i, j, X, increase, theta, clip_min, clip_max):
    # perturb our input sample
    if increase:
        X[0, i] = np.minimum(clip_max, X[0, i] + theta)
        X[0, j] = np.minimum(clip_max, X[0, j] + theta)
    else:
        X[0, i] = np.maximum(clip_min, X[0, i] - theta)
        X[0, j] = np.maximum(clip_min, X[0, j] - theta)

    return X


def saliency_map(grads_target, grads_other, search_domain, increase):
    # Compute the size of the input (the number of features)
    nf = len(grads_target)

    # Remove the already-used input features from the search space
    invalid = list(set(range(nf)) - search_domain)
    grads_target[invalid] = 0
    grads_other[invalid] = 0

    # Create a 2D numpy array of the sum of grads_target and grads_other
    target_sum = grads_target.reshape((1, nf)) + grads_target.reshape((nf, 1))
    other_sum = grads_other.reshape((1, nf)) + grads_other.reshape((nf, 1))

    # Create a mask to only keep features that match saliency map conditions
    if increase:
        scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
        scores_mask = ((target_sum < 0) & (other_sum > 0))

    # Create a 2D numpy array of the scores for each pair of candidate features
    scores = scores_mask * (-target_sum * other_sum)

    # A pixel can only be selected (and changed) once
    np.fill_diagonal(scores, 0)

    # Extract the best two pixels
    best = np.argmax(scores)
    p1, p2 = best % nf, best // nf

    # Remove used pixels from our search domain
    search_domain.remove(p1)
    search_domain.remove(p2)

    return p1, p2, search_domain


def jacobian(sess, x, grads, target, X, nb_features, nb_classes):
    # Prepare feeding dictionary for all gradient computations
    feed_dict = {x: X}

    # Initialize a numpy array to hold the Jacobian component values
    jacobian_val = np.zeros((nb_classes, nb_features), dtype=np.float32)

    # Compute the gradients for all classes
    for class_ind, grad in enumerate(grads):
        run_grad = sess.run(grad, feed_dict)
        jacobian_val[class_ind] = np.reshape(run_grad, (1, nb_features))

    # Sum over all classes different from the target class to prepare for
    # saliency map computation in the next step of the attack
    other_classes = utils.other_classes(nb_classes, target)
    grad_others = np.sum(jacobian_val[other_classes, :], axis=0)

    return jacobian_val[target], grad_others


def jacobian_graph(predictions, x, nb_classes):
    # This function will return a list of TF gradients
    list_derivatives = []

    # Define the TF graph elements to compute our derivatives for each class
    for class_ind in xrange(nb_classes):
        derivatives, = tf.gradients(predictions[:, class_ind], x)
        list_derivatives.append(derivatives)

    return list_derivatives


def jsma(sess, x, predictions, grads, sample, target, theta, gamma, clip_min, clip_max):
    # Copy the source sample and define the maximum number of features
    # (i.e. the maximum number of iterations) that we may perturb
    adv_x = copy.copy(sample)
    # count the number of features. For MNIST, 1x28x28 = 784; for
    # CIFAR, 3x32x32 = 3072; etc.
    nb_features = np.product(adv_x.shape[1:])
    # reshape sample for sake of standardization
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    # compute maximum number of iterations
    max_iters = np.floor(nb_features * gamma / 2)

    # Find number of classes based on grads
    nb_classes = len(grads)

    increase = bool(theta > 0)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] > clip_min])

    # Initialize the loop variables
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    current = helpers.model_argmax(sess, x, predictions, adv_x_original_shape)

    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters and
           len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, nb_classes)

        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)

        # Update our current prediction by querying the model
        current = helpers.model_argmax(sess, x, predictions,
                                        adv_x_original_shape)

        # Update loop variables
        iteration = iteration + 1

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


def jsma_batch(sess, x, pred, grads, X, theta, gamma, clip_min, clip_max, nb_classes, targets=None):
    X_adv = np.zeros(X.shape)

    for ind, val in enumerate(X):
        val = np.expand_dims(val, axis=0)
        if targets is None:
            # No targets provided, randomly choose from other classes
            from helpers import model_argmax
            gt = model_argmax(sess, x, pred, val)

            # Randomly choose from the incorrect classes for each sample
            from .utils import random_targets
            target = random_targets(gt, nb_classes)[0]
        else:
            target = targets[ind]

        X_adv[ind], _, _ = jsma(sess, x, pred, grads, val, np.argmax(target),
                                theta, gamma, clip_min, clip_max)

    return np.asarray(X_adv, dtype=np.float32)


def jacobian_augmentation(sess, x, X_sub_prev, Y_sub, grads, lmbda, keras_phase=None):
    assert len(x.get_shape()) == len(np.shape(X_sub_prev))
    assert len(grads) >= np.max(Y_sub) + 1
    assert len(X_sub_prev) == len(Y_sub)

    if keras_phase is not None:
        warnings.warn("keras_phase argument is deprecated and will be removed"
                      " on 2017-09-28. Instead, use K.set_learning_phase(0) at"
                      " the start of your script and serve with tensorflow.")

    # Prepare input_shape (outside loop) for feeding dictionary below
    input_shape = list(x.get_shape())
    input_shape[0] = 1

    # Create new numpy array for adversary training data
    # with twice as many components on the first dimension.
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, input in enumerate(X_sub_prev):
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Prepare feeding dictionary
        feed_dict = {x: np.reshape(input, input_shape)}

        # Compute sign matrix
        grad_val = sess.run([tf.sign(grad)], feed_dict=feed_dict)[0]

        # Create new synthetic point in adversary substitute training set
        X_sub[2*ind] = X_sub[ind] + lmbda * grad_val

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


def deepfool_batch(sess, x, pred, logits, grads, X, nb_candidate, overshoot,
				   max_iter, clip_min, clip_max, nb_classes, feed=None):
	X_adv = deepfool_attack(sess, x, pred, logits, grads, X, nb_candidate,
							overshoot, max_iter, clip_min, clip_max, feed=feed)
	return np.asarray(X_adv, dtype=np.float32)


def deepfool_attack(sess, x, predictions, logits, grads, sample, nb_candidate,
					overshoot, max_iter, clip_min, clip_max, feed=None):
	adv_x = copy.copy(sample)
	iteration = 0
	current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
	if current.shape == ():
		current = np.array([current])
	w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
	r_tot = np.zeros(sample.shape)
	original = current  # use original label as the reference

	while (np.any(current == original) and iteration < max_iter):

		if iteration % 5 == 0 and iteration > 0:
			print("Attack result at iteration %d is %d"%(iteration,current))
		gradients = sess.run(grads, feed_dict={x: adv_x})
		predictions_val = sess.run(predictions, feed_dict={x: adv_x})
		for idx in range(sample.shape[0]):
			pert = np.inf
			if current[idx] != original[idx]:
				continue
			for k in range(1, nb_candidate):
				w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
				f_k = predictions_val[idx, k] - predictions_val[idx, 0]

				pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
				if pert_k < pert:
					pert = pert_k
					w = w_k
			r_i = pert*w/np.linalg.norm(w)
			r_tot[idx, ...] = r_tot[idx, ...] + r_i

		adv_x = np.clip(r_tot + sample, clip_min, clip_max)
		current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
		if current.shape == ():
			current = np.array([current])
		iteration = iteration + 1

	print("Attack result at iteration %d is %d"%(iteration, current))
	print("%d out of %d"%(sum(current != original), sample.shape[0]) + " becomes adversarial examples at iteration %d"%(iteration))
	adv_x = np.clip((1+overshoot)*r_tot + sample, clip_min, clip_max)
	return adv_x


def vatm(model, x, logits, eps, num_iterations=1, xi=1e-6, clip_min=None, clip_max=None, scope=None):
	"""
	:param eps: the epsilon (input variation parameter)
	:param num_iterations: the number of iterations
	:param xi: the finite difference parameter
	"""
	with tf.name_scope(scope, "virtual_adversarial_perturbation"):
		d = tf.random_normal(tf.shape(x))
		for i in range(num_iterations):
			d = xi * utils_tf.l2_batch_normalize(d)
			logits_d = model.get_logits(x + d)
			kl = utils_tf.kl_with_logits(logits, logits_d)
			Hd = tf.gradients(kl, d)[0]
			d = tf.stop_gradient(Hd)
		d = eps * utils_tf.l2_batch_normalize(d)
		adv_x = x + d
		if (clip_min is not None) and (clip_max is not None):
			adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
		return adv_x
