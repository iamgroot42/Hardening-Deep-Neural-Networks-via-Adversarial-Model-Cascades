from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def tf_model_loss(y, model):
	"""
	Define loss of TF graph
	Returns vector with per sample loss
	"""
	op = model.op
	if "softmax" in str(op).lower():
		logits, = op.inputs
	else:
		logits = model

	out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
	return out


def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
	"""
	TensorFlow implementation of the Fast Gradient Sign method.
	Returns a tensor for the adversarial example.
	"""
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
