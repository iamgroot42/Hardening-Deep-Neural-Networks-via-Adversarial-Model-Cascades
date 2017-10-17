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

		#if iteration % 5 == 0 and iteration > 0:
		#	print("Attack result at iteration %d is "%(iteration) + str(current))
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

	#print("Attack result at iteration %d is %d"%(iteration, current))
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


class ElasticNetMethod(object):

	def __init__(self, sess, model, beta,
				 batch_size, confidence,
				 targeted, learning_rate,
				 binary_search_steps, max_iterations,
				 abort_early, initial_const,
				 clip_min, clip_max, num_labels, shape):
		"""
		EAD Attack with the EN Decision Rule
		Return a tensor that constructs adversarial examples for the given
		input. Generate uses tf.py_func in order to operate over tensors.
		:param sess: a TF session.
		:param model: a cleverhans.model.Model object.
		:param beta: Trades off L2 distortion with L1 distortion: higher
					 produces examples with lower L1 distortion, at the
					 cost of higher L2 (and typically Linf) distortion
		:param confidence: Confidence of adversarial examples: higher produces
						   examples with larger l2 distortion, but more
						   strongly classified as adversarial.
		:param targeted: boolean controlling the behavior of the adversarial
						 examples produced. If set to False, they will be
						 misclassified in any wrong class. If set to True,
						 they will be misclassified in a chosen target class.
		:param learning_rate: The learning rate for the attack algorithm.
							  Smaller values produce better results but are
							  slower to converge.
		:param binary_search_steps: The number of times we perform binary
									search to find the optimal tradeoff-
									constant between norm of the perturbation
									and confidence of the classification.
		:param max_iterations: The maximum number of iterations. Setting this
							   to a larger value will produce lower distortion
							   results. Using only a few iterations requires
							   a larger learning rate, and will produce larger
							   distortion results.
		:param abort_early: If true, allows early abort when the total
							loss starts to increase (greatly speeds up attack,
							but hurts performance, particularly on ImageNet)
		:param initial_const: The initial tradeoff-constant to use to tune the
							  relative importance of size of the perturbation
							  and confidence of classification.
							  If binary_search_steps is large, the initial
							  constant is not important. A smaller value of
							  this constant gives lower distortion results.
		:param num_labels: the number of classes in the model's output.
		:param shape: the shape of the model's input tensor.
		"""

		self.sess = sess
		self.TARGETED = targeted
		self.LEARNING_RATE = learning_rate
		self.MAX_ITERATIONS = max_iterations
		self.BINARY_SEARCH_STEPS = binary_search_steps
		self.ABORT_EARLY = abort_early
		self.CONFIDENCE = confidence
		self.initial_const = initial_const
		self.batch_size = batch_size
		self.clip_min = clip_min
		self.clip_max = clip_max
		self.model = model

		self.beta = beta
		self.beta_t = tf.cast(self.beta, tf.float32)

		self.repeat = binary_search_steps >= 10

		self.shape = shape = tuple([batch_size] + list(shape))

		# these are variables to be more efficient in sending data to tf
		self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='timg')
		self.newimg = tf.Variable(np.zeros(shape), dtype=tf.float32, name='newimg')
		self.slack = tf.Variable(np.zeros(shape), dtype=tf.float32, name='slack')
		self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32, name='tlab')
		self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='const')

		# and here's what we use to assign them
		self.assign_timg = tf.placeholder(tf.float32, shape, name='assign_timg')
		self.assign_newimg = tf.placeholder(tf.float32, shape, name='assign_newimg')
		self.assign_slack = tf.placeholder(tf.float32, shape, name='assign_slack')
		self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels), name='assign_tlab')
		self.assign_const = tf.placeholder(tf.float32, [batch_size], name='assign_const')

		self.global_step = tf.Variable(0, trainable=False)
		self.global_step_t = tf.cast(self.global_step, tf.float32)

		"""Fast Iterative Shrinkage Thresholding"""
		"""--------------------------------"""
		self.zt = tf.divide(self.global_step_t, self.global_step_t + tf.cast(3, tf.float32))

		cond1 = tf.cast(tf.greater(tf.subtract(self.slack, self.timg), self.beta_t), tf.float32)
		cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(self.slack, self.timg)), self.beta_t), tf.float32)
		cond3 = tf.cast(tf.less(tf.subtract(self.slack, self.timg), tf.negative(self.beta_t)), tf.float32)

		upper = tf.minimum(tf.subtract(self.slack, self.beta_t), tf.cast(self.clip_max, tf.float32))
		lower = tf.maximum(tf.add(self.slack, self.beta_t), tf.cast(self.clip_min, tf.float32))

		self.assign_newimg = tf.multiply(cond1, upper)
		self.assign_newimg += tf.multiply(cond2, self.timg)
		self.assign_newimg += tf.multiply(cond3, lower)

		self.assign_slack = self.assign_newimg
		self.assign_slack += tf.multiply(self.zt, self.assign_newimg - self.newimg)

		self.setter = tf.assign(self.newimg, self.assign_newimg)
		self.setter_y = tf.assign(self.slack, self.assign_slack)
		"""--------------------------------"""

		# prediction BEFORE-SOFTMAX of the model
		self.output = model.get_logits(self.newimg)
		self.output_y = model.get_logits(self.slack)

		# distance to the input data
		self.l2dist = tf.reduce_sum(tf.square(self.newimg-self.timg), list(range(1, len(shape))))
		self.l2dist_y = tf.reduce_sum(tf.square(self.slack-self.timg), list(range(1, len(shape))))
		self.l1dist = tf.reduce_sum(tf.abs(self.newimg-self.timg), list(range(1, len(shape))))
		self.l1dist_y = tf.reduce_sum(tf.abs(self.slack-self.timg), list(range(1, len(shape))))
		self.elasticdist = self.l2dist + tf.multiply(self.l1dist, self.beta_t)
		self.elasticdist_y = self.l2dist_y + tf.multiply(self.l1dist_y, self.beta_t)

		# compute the probability of the label class versus the maximum other

		real = tf.reduce_sum((self.tlab) * self.output, 1)
		real_y = tf.reduce_sum((self.tlab) * self.output_y, 1)
		other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)
		other_y = tf.reduce_max((1 - self.tlab) * self.output_y - (self.tlab * 10000), 1)

		if self.TARGETED:
			# if targeted, optimize for making the other class most likely
			loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
			loss1_y = tf.maximum(0.0, other_y - real_y + self.CONFIDENCE)
		else:
			# if untargeted, optimize for making this class least likely.
			loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)
			loss1_y = tf.maximum(0.0, real_y - other_y + self.CONFIDENCE)

		# sum up the losses
		self.loss21 = tf.reduce_sum(self.l1dist)
		self.loss21_y = tf.reduce_sum(self.l1dist_y)
		self.loss2 = tf.reduce_sum(self.l2dist)
		self.loss2_y = tf.reduce_sum(self.l2dist_y)
		self.loss1 = tf.reduce_sum(self.const * loss1)
		self.loss1_y = tf.reduce_sum(self.const * loss1_y)
		self.loss2 = tf.reduce_sum(self.l2dist)

		self.loss_opt = self.loss1_y+self.loss2_y
		self.loss = self.loss1+self.loss2+tf.multiply(self.beta_t, self.loss21)

		self.learning_rate = tf.train.polynomial_decay(self.LEARNING_RATE,
								   self.global_step,
								   self.MAX_ITERATIONS,
								   0, power=0.5)

		# Setup the optimizer and keep track of variables we're creating
		start_vars = set(x.name for x in tf.global_variables())
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		self.train = optimizer.minimize(self.loss_opt,
						var_list=[self.slack],
						global_step=self.global_step)
		end_vars = tf.global_variables()
		new_vars = [x for x in end_vars if x.name not in start_vars]

		# these are the variables to initialize when we run
		self.setup = []
		self.setup.append(self.timg.assign(self.assign_timg))
		self.setup.append(self.tlab.assign(self.assign_tlab))
		self.setup.append(self.const.assign(self.assign_const))

		self.init = tf.variables_initializer(var_list=[self.global_step] +
							[self.slack] + [self.newimg] +
							new_vars)

	def attack(self, imgs, targets):
		"""
		Perform the EAD attack on the given instance for the given targets.
		If self.targeted is true, then the targets represents the target labels
		If self.targeted is false, then targets are the original class labels
		"""
		r = []
		for i in range(0, len(imgs), self.batch_size):
			# OOPS PROBLEM, FIX IN CLEVERHANS, FIX IN PR
			if(i + self.batch_size >= len(imgs)):
				break
			print("Running EAD attack on instance " + str(i) + " of " + str(len(imgs)))
			r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
		return np.array(r)


	def attack_batch(self, imgs, labs):
		def compare(x, y):
			if not isinstance(x, (float, int, np.int64)):
				x = np.copy(x)
				if self.TARGETED:
					x[y] -= self.CONFIDENCE
				else:
					x[y] += self.CONFIDENCE
				x = np.argmax(x)
			if self.TARGETED:
				return x == y
			else:
				return x != y

		batch_size = self.batch_size

		imgs = np.clip(imgs, self.clip_min, self.clip_max)

		lower_bound = np.zeros(batch_size)
		CONST = np.ones(batch_size) * self.initial_const
		upper_bound = np.ones(batch_size) * 1e10

		o_besten = [1e10] * batch_size
		o_bestscore = [-1] * batch_size
		o_bestattack = np.copy(imgs)

		for outer_step in range(self.BINARY_SEARCH_STEPS):
			self.sess.run(self.init)
			batch = imgs[:batch_size]
			batchlab = labs[:batch_size]

			besten = [1e10] * batch_size
			bestscore = [-1] * batch_size
			print("Binary search step %d of %d"%(outer_step, self.BINARY_SEARCH_STEPS))

			if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
				CONST = upper_bound

			self.sess.run(self.setup, {self.assign_timg: batch,
									   self.assign_tlab: batchlab,
									   self.assign_const: CONST})
			self.sess.run(self.setter, feed_dict={self.assign_newimg: batch})
			self.sess.run(self.setter_y, feed_dict={self.assign_slack: batch})
			prev = 1e6
			for iteration in range(self.MAX_ITERATIONS):
				self.sess.run([self.train])
				self.sess.run([self.setter, self.setter_y])
				l, l2s, l1s, elastic = self.sess.run([self.loss,
									 self.l2dist,
									 self.l1dist,
									 self.elasticdist])
				scores, nimg = self.sess.run([self.output, self.newimg])

				#if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
					#_logger.debug(("    Iteration {} of {}: loss={:.3g} " +
					#			   "l2={:.3g} l1={:.3g} f={:.3g}")
					#			  .format(iteration, self.MAX_ITERATIONS,
					#					  l, np.mean(l2s), np.mean(l1s),
					#					  np.mean(scores)))

				if self.ABORT_EARLY and iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
					if l > prev * .9999:
						print(" Failed to make progress; stop early")
						break
					prev = l

				for e, (en, sc, ii) in enumerate(zip(elastic, scores, nimg)):
					lab = np.argmax(batchlab[e])
					if en < besten[e] and compare(sc, lab):
						besten[e] = en
						bestscore[e] = np.argmax(sc)
					if en < o_besten[e] and compare(sc, lab):
						o_besten[e] = en
						o_bestscore[e] = np.argmax(sc)
						o_bestattack[e] = ii

			for e in range(batch_size):
				if compare(bestscore[e], np.argmax(batchlab[e])) and \
				   bestscore[e] != -1:
					upper_bound[e] = min(upper_bound[e], CONST[e])
					if upper_bound[e] < 1e9:
						CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
				else:
					lower_bound[e] = max(lower_bound[e], CONST[e])
					if upper_bound[e] < 1e9:
						CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
					else:
						CONST[e] *= 10
			print(" Successfully generated adversarial examples on %d of %d instances."%(sum(upper_bound < 1e9), batch_size))
			o_besten = np.array(o_besten)
			mean = np.mean(np.sqrt(o_besten[o_besten < 1e9]))
			print(" Elastic Mean successful distortion: " + str(mean))

		o_besten = np.array(o_besten)
		return o_bestattack

