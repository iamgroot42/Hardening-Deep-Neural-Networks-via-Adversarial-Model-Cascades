import common

import math
import numpy as np
import os
import keras
from keras.backend import categorical_crossentropy
import six
import tensorflow as tf
import time

from utils import batch_indices

def tf_model_loss(y, model, mean=True):
	op = model.op
	if "softmax" in str(op).lower():
		logits, = op.inputs
	else:
		logits = model
	out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	return out


def tf_model_train(sess, x, y, predictions, X_train, Y_train, save=False,
				   predictions_adv=None, verbose=True):
	# Define loss
	loss = tf_model_loss(y, predictions)
	if predictions_adv is not None:
		loss = (loss + tf_model_loss(y, predictions_adv)) / 2

	train_step = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08).minimize(loss)
	# train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
	with sess.as_default():
		init = tf.global_variables_initializer()
		# init = tf.initialize_all_variables()
		sess.run(init)

		for epoch in six.moves.xrange(FLAGS.nb_epochs):
			if verbose:
				print("Epoch " + str(epoch))

			# Compute number of batches
			nb_batches = int(math.ceil(float(len(X_train)) / FLAGS.batch_size))
			assert nb_batches * FLAGS.batch_size >= len(X_train)

			prev = time.time()
			for batch in range(nb_batches):

				# Compute batch start and end indices
				start, end = batch_indices(batch, len(X_train), FLAGS.batch_size)

				# Perform one training step
				train_step.run(feed_dict={x: X_train[start:end],
										  y: Y_train[start:end],
										  keras.backend.learning_phase(): 1})
			assert end >= len(X_train) # Check that all examples were used
			cur = time.time()
			if verbose:
				print("\tEpoch took " + str(cur - prev) + " seconds")
			prev = cur

		if save:
			save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
			saver = tf.train.Saver()
			saver.save(sess, save_path)
			if verbose:
				print("Completed model training and model saved at:" + str(save_path))
		else:
			if verbose:
				print("Completed model training.")

	return True


def tf_model_eval(sess, x, y, model, X_test, Y_test, verbose=True):
	acc_value = keras.metrics.categorical_accuracy(y, model)
	accuracy = 0.0
	batch_size = 32
	with sess.as_default():
		nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
		assert nb_batches * batch_size >= len(X_test)
		for batch in range(nb_batches):
			if batch % 100 == 0 and batch > 0:
				if verbose:
					print("Batch " + str(batch))
			# Must not use the `batch_indices` function here, because it
			# repeats some examples.
			start = batch * batch_size
			end = min(len(X_test), start + batch_size)
			cur_batch_size = end - start
			# The last batch may be smaller than all others, so we need to
			# account for variable batch size here
			accuracy += cur_batch_size * acc_value.eval(feed_dict={x: X_test[start:end],
											y: Y_test[start:end],
											keras.backend.learning_phase(): 0})
		assert end >= len(X_test)
		accuracy /= len(X_test)
	return accuracy


def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, verbose=True):
	n = len(numpy_inputs)
	batch_size=32
	assert n > 0
	assert n == len(tf_inputs)
	m = numpy_inputs[0].shape[0]
	for i in six.moves.xrange(1, n):
		assert numpy_inputs[i].shape[0] == m
	out = []
	for _ in tf_outputs:
		out.append([])
	with sess.as_default():
		for start in six.moves.xrange(0, m, batch_size):
			batch = start // batch_size
			if batch % 100 == 0 and batch > 0:
				if verbose:
					print("Batch " + str(batch+1))

			start = batch * batch_size
			end = start + batch_size
			numpy_input_batches = [numpy_input[start:end] for numpy_input in numpy_inputs]
			cur_batch_size = numpy_input_batches[0].shape[0]
			assert cur_batch_size <= batch_size
			for e in numpy_input_batches:
				assert e.shape[0] == cur_batch_size

			feed_dict = dict(zip(tf_inputs, numpy_input_batches))
			feed_dict[keras.backend.learning_phase()] = 0
			numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
			for e in numpy_output_batches:
				assert e.shape[0] == cur_batch_size, e.shape
			for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
				out_elem.append(numpy_output_batch)

	out = [np.concatenate(x, axis=0) for x in out]
	for e in out:
		assert e.shape[0] == m, e.shape
	return out


def l2_batch_normalize(x, epsilon=1e-12, scope=None):
    """
    Helper function to normalize a batch of vectors.
    :param x: the input placeholder
    :param epsilon: stabilizes division
    :return: the batch of l2 normalized vector
    """
    with tf.name_scope(scope, "l2_batch_normalize") as scope:
        x_shape = tf.shape(x)
        x = tf.contrib.layers.flatten(x)
        x /= (epsilon + tf.reduce_max(tf.abs(x), 1, keep_dims=True))
        square_sum = tf.reduce_sum(tf.square(x), 1, keep_dims=True)
        x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
        x_norm = tf.multiply(x, x_inv_norm)
        return tf.reshape(x_norm, x_shape, scope)


def kl_with_logits(p_logits, q_logits, scope=None,
                   loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
    """Helper function to compute kl-divergence KL(p || q)
    """
    with tf.name_scope(scope, "kl_divergence") as name:
        p = tf.nn.softmax(p_logits)
        p_log = tf.nn.log_softmax(p_logits)
        q_log = tf.nn.log_softmax(q_logits)
        loss = tf.reduce_mean(tf.reduce_sum(p * (p_log - q_log), axis=1),
                              name=name)
        tf.losses.add_loss(loss, loss_collection)
        return loss
