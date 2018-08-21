import math
import numpy as np
import os
import keras
from keras.backend import categorical_crossentropy
import six
import tensorflow as tf
import time

def batch_indices(batch_nb, data_length, batch_size):
	start = int(batch_nb * batch_size)
	end = int((batch_nb + 1) * batch_size)

	if end > data_length:
		shift = end - data_length
		start -= shift
		end -= shift

	return start, end

def tf_model_loss(y, model, mean=True):
	op = model.op
	if "softmax" in str(op).lower():
		logits, = op.inputs
	else:
		logits = model
	out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
	if mean:
		out = tf.reduce_mean(out)
	return out


def tf_model_train(sess, model, x, y, predictions, X_train, Y_train, FLAGS, evaluate, scheduler, data_generator,
				   predictions_adv=None, verbose=True, optimizer=None):

	# Define loss
	loss = tf_model_loss(y, predictions)
	if predictions_adv is not None:
		print("Using lambda = " + str(FLAGS.eta))
		loss = (loss + FLAGS.eta * tf_model_loss(y, predictions_adv)) / 2

	lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
	update_ops = model.updates

	if optimizer is None:
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=0.9, use_nesterov=True)
	else:
		if not isinstance(optimizer, tf.train.Optimizer):
			raise ValueError("optimizer object must be from a child class of "
				"tf.train.Optimizer")

	with tf.control_dependencies(update_ops):
		train_step = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=0.9, use_nesterov=True).minimize(loss)

	with sess.as_default():
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		for epoch in six.moves.xrange(FLAGS.nb_epochs):
			iterator = data_generator.flow(X_train, Y_train, batch_size=FLAGS.batch_size)
			learning_rate = scheduler(epoch)
			# Compute number of batches
			nb_batches = int(math.ceil(float(len(X_train)) / FLAGS.batch_size))
			assert nb_batches * FLAGS.batch_size >= len(X_train)
			avg_loss = 0
			prev = time.time()
			for batch in range(nb_batches):
				# Compute batch start and end indices
				(currX, currY) = next(iterator)
				train_step.run(feed_dict={x:currX, y:currY, keras.backend.learning_phase():1, lr_placeholder:learning_rate})
				l = sess.run(loss, feed_dict={x: currX, y: currY, keras.backend.learning_phase(): 0})
				avg_loss += l / nb_batches
			cur = time.time()
			val_acc = evaluate()
			if verbose:
				print("Epoch " + str(epoch) + " took " + str(cur - prev) + " seconds, lr = " + str(learning_rate) + ", validation accuracy is " + str(val_acc) + ", loss is " + str(avg_loss))
			prev = cur

		if verbose:
			print("Completed model training.")

	return True

def model_argmax(sess, x, predictions, samples, feed=None):
	feed_dict = {x: samples}
	if feed is not None:
		feed_dict.update(feed)
	probabilities = sess.run(predictions, feed_dict)

	if samples.shape[0] == 1:
		return np.argmax(probabilities)
	else:
		return np.argmax(probabilities, axis=1)

def model_eval(sess, x, y, predictions, args, X_test=None, Y_test=None, feed=None):
	correct_preds = tf.equal(tf.argmax(y, axis=-1), tf.argmax(predictions, axis=-1))
	accuracy = 0.0

	with sess.as_default():
		# Compute number of batches
		nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
		assert nb_batches * args.batch_size >= len(X_test)

		X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
						 dtype=X_test.dtype)
		Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
						 dtype=Y_test.dtype)
		for batch in range(nb_batches):
			# Must not use the `batch_indices` function here, because it
			# repeats some examples.
			# It's acceptable to repeat during training, but not eval.
			start = batch * args.batch_size
			end = min(len(X_test), start + args.batch_size)

			# The last batch may be smaller than all others. This should not
			# affect the accuarcy disproportionately.
			cur_batch_size = end - start
			X_cur[:cur_batch_size] = X_test[start:end]
			Y_cur[:cur_batch_size] = Y_test[start:end]
			feed_dict = {x: X_cur, y: Y_cur, keras.backend.learning_phase(): 0}
			if feed is not None:
				feed_dict.update(feed)
			cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

			accuracy += cur_corr_preds[:cur_batch_size].sum()
		assert end >= len(X_test)

		# Divide by number of examples to get final value
		accuracy /= len(X_test)

	return accuracy
