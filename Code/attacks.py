from abc import ABCMeta
import numpy as np
import warnings

import tensorflow as tf
import attacks_tf

class Attack:
	__metaclass__ = ABCMeta

	def __init__(self, model, back='tf', sess=None):
		if not(back == 'tf' or back == 'th'):
			raise ValueError("Backend argument must either be 'tf' or 'th'.")
		if back == 'tf' and sess is None:
			raise Exception("A tf session was not provided in sess argument.")
		if back == 'th' and sess is not None:
			raise Exception("A session should not be provided when using th.")
		if not hasattr(model, '__call__'):
			raise ValueError("model argument must be a function that returns "
							 "the symbolic output when given an input tensor.")
		if back == 'th':
			warnings.warn("cleverhans support for Theano is deprecated and "
						  "will be dropped on 2017-11-08.")

		# Prepare attributes
		self.model = model
		self.back = back
		self.sess = sess
		self.inf_loop = False

	def generate(self, x, **kwargs):
		if self.back == 'th':
			raise NotImplementedError('Theano version not implemented.')

		if not self.inf_loop:
			self.inf_loop = True
			assert self.parse_params(**kwargs)
			import tensorflow as tf
			graph = tf.py_func(self.generate_np, [x], tf.float32)
			self.inf_loop = False
			return graph
		else:
			error = "No symbolic or numeric implementation of attack."
			raise NotImplementedError(error)

	def generate_np(self, x_val, **kwargs):
		if self.back == 'th':
			raise NotImplementedError('Theano version not implemented.')

		if not self.inf_loop:
			self.inf_loop = True
			import tensorflow as tf

			# Generate this attack's graph if not done previously
			if not hasattr(self, "_x") and not hasattr(self, "_x_adv"):
				input_shape = list(x_val.shape)
				input_shape[0] = None
				self._x = tf.placeholder(tf.float32, shape=input_shape)
				self._x_adv = self.generate(self._x, **kwargs)
			self.inf_loop = False
		else:
			error = "No symbolic or numeric implementation of attack."
			raise NotImplementedError(error)

		return self.sess.run(self._x_adv, feed_dict={self._x: x_val})

	def parse_params(self, params=None):
		return True


class BasicIterativeMethod(Attack):
	def __init__(self, model, back='tf', sess=None):
		"""
		Create a BasicIterativeMethod instance.
		"""
		super(BasicIterativeMethod, self).__init__(model, back, sess)

	def generate(self, x, **kwargs):
		import tensorflow as tf

		# Parse and save attack-specific parameters
		assert self.parse_params(**kwargs)

		# Initialize loop variables
		eta = 0

		# Fix labels to the first model predictions for loss computation
		model_preds = self.model(x)
		preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
		y = tf.to_float(tf.equal(model_preds, preds_max))
		fgsm_params = {'eps': self.eps_iter, 'y': y, 'ord': self.ord}

		for i in range(self.nb_iter):
			FGSM = FastGradientMethod(self.model, back=self.back,
									  sess=self.sess)
			# Compute this step's perturbation
			eta = FGSM.generate(x + eta, **fgsm_params) - x

			# Clipping perturbation eta to self.ord norm ball
			if self.ord == np.inf:
				eta = tf.clip_by_value(eta, -self.eps, self.eps)
			elif self.ord in [1, 2]:
				reduc_ind = list(xrange(1, len(eta.get_shape())))
				if self.ord == 1:
					norm = tf.reduce_sum(tf.abs(eta),
										 reduction_indices=reduc_ind,
										 keep_dims=True)
				elif self.ord == 2:
					norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
												 reduction_indices=reduc_ind,
												 keep_dims=True))
				eta = eta * self.eps / norm

		# Define adversarial example (and clip if necessary)
		adv_x = x + eta
		if self.clip_min is not None and self.clip_max is not None:
			adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

		return adv_x

	def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
					 ord=np.inf, clip_min=None, clip_max=None, **kwargs):

		# Save attack-specific parameters
		self.eps = eps
		self.eps_iter = eps_iter
		self.nb_iter = nb_iter
		self.y = y
		self.ord = ord
		self.clip_min = clip_min
		self.clip_max = clip_max

		# Check if order of the norm is acceptable given current implementation
		if self.ord not in [np.inf, 1, 2]:
			raise ValueError("Norm order must be either np.inf, 1, or 2.")
		if self.back == 'th':
			error_string = "BasicIterativeMethod is not implemented in Theano"
			raise NotImplementedError(error_string)

		return True


class SaliencyMapMethod(Attack):
	def __init__(self, model, back='tf', sess=None):
		"""
		Create a SaliencyMapMethod instance.
		"""
		super(SaliencyMapMethod, self).__init__(model, back, sess)

		if self.back == 'th':
			error = "Theano version of SaliencyMapMethod not implemented."
			raise NotImplementedError(error)

	def generate(self, x, **kwargs):
		"""
		Attack-specific parameters:
		"""
		from attacks_tf import jacobian_graph, jsma_batch

		# Parse and save attack-specific parameters
		assert self.parse_params(**kwargs)

		# Define Jacobian graph wrt to this input placeholder
		preds = self.model(x)
		grads = jacobian_graph(preds, x, self.nb_classes)

		# Define appropriate graph (targeted / random target labels)
		if self.targets is not None:
			def jsma_wrap(x_val, targets):
				return jsma_batch(self.sess, x, preds, grads, x_val,
								  self.theta, self.gamma, self.clip_min,
								  self.clip_max, self.nb_classes,
								  targets=targets)

			# Attack is targeted, target placeholder will need to be fed
			wrap = tf.py_func(jsma_wrap, [x, self.targets], tf.float32)
		else:
			def jsma_wrap(x_val):
				return jsma_batch(self.sess, x, preds, grads, x_val,
								  self.theta, self.gamma, self.clip_min,
								  self.clip_max, self.nb_classes,
								  targets=None)

			# Attack is untargeted, target values will be chosen at random
			wrap = tf.py_func(jsma_wrap, [x], tf.float32)

		return wrap

	def generate_np(self, x_val, **kwargs):
		import tensorflow as tf
		# Generate this attack's graph if it hasn't been done previously
		if not hasattr(self, "_x"):
			input_shape = list(x_val.shape)
			input_shape[0] = None
			self._x = tf.placeholder(tf.float32, shape=input_shape)
			self._x_adv = self.generate(self._x, **kwargs)

		# Run symbolic graph without or with true labels
		if 'y_val' not in kwargs or kwargs['y_val'] is None:
			feed_dict = {self._x: x_val}
		else:
			if self.targets is None:
				raise Exception("This attack was instantiated untargeted.")
			else:
				if len(kwargs['y_val'].shape) > 1:
					nb_targets = len(kwargs['y_val'])
				else:
					nb_targets = 1
				if nb_targets != len(x_val):
					raise Exception("Specify exactly one target per input.")
			feed_dict = {self._x: x_val, self.targets: kwargs['y_val']}
		return self.sess.run(self._x_adv, feed_dict=feed_dict)

	def parse_params(self, theta=1., gamma=np.inf, nb_classes=10, clip_min=0.,
					 clip_max=1., targets=None, **kwargs):
		self.theta = theta
		self.gamma = gamma
		self.nb_classes = nb_classes
		self.clip_min = clip_min
		self.clip_max = clip_max
		self.targets = targets

		return True


def jsma(sess, x, predictions, grads, sample, target, theta, gamma=np.inf,
		 increase=True, back='tf', clip_min=None, clip_max=None):
	warnings.warn("attacks.jsma is deprecated and will be removed on "
				  "2017-09-27. Instantiate an object from SaliencyMapMethod.")
	if back == 'tf':
		# Compute Jacobian-based saliency map attack using TensorFlow
		from attacks_tf import jsma
		return jsma(sess, x, predictions, grads, sample, target, theta, gamma,
					clip_min, clip_max)
	elif back == 'th':
		raise NotImplementedError("Theano jsma not implemented.")


class DeepFool(Attack):
	def __init__(self, model, back='tf', sess=None):
		super(DeepFool, self).__init__(model, back, sess)

		self.structural_kwargs = ['over_shoot', 'max_iter', 'clip_max',
								  'clip_min', 'nb_candidate']

	def generate(self, x, **kwargs):
		assert self.parse_params(**kwargs)

		logits = self.model.get_logits(x)
		self.nb_classes = logits.get_shape().as_list()[-1]
		assert self.nb_candidate <= self.nb_classes,\
			'nb_candidate should not be greater than nb_classes'
		preds = tf.reshape(tf.nn.top_k(logits, k=self.nb_candidate)[0],
						   [-1, self.nb_candidate])

		grads = tf.stack(attacks_tf.jacobian_graph(preds, x, self.nb_candidate), axis=1)

		def deepfool_wrap(x_val):
			return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
								  self.nb_candidate, self.overshoot,
								  self.max_iter, self.clip_min, self.clip_max,
								  self.nb_classes)
		return tf.py_func(deepfool_wrap, [x], tf.float32)

	def parse_params(self, nb_candidate=10, overshoot=0.02, max_iter=50,
					 nb_classes=None, clip_min=0., clip_max=1., **kwargs):
		"""
		:param nb_candidate: The number of classes to test against, i.e.,
							 deepfool only consider nb_candidate classes when
							 attacking(thus accelerate speed). The nb_candidate
							 classes are chosen according to the prediction
							 confidence during implementation.
		:param overshoot: A termination criterion to prevent vanishing updates
		:param max_iter: Maximum number of iteration for deepfool
		:param nb_classes: The number of model output classes
		:param clip_min: Minimum component value for clipping
		:param clip_max: Maximum component value for clipping
		"""
		self.nb_candidate = nb_candidate
		self.overshoot = overshoot
		self.max_iter = max_iter
		self.clip_min = clip_min
		self.clip_max = clip_max
		return True




class VirtualAdversarialMethod(Attack):
    def __init__(self, model, back='tf', sess=None):
        super(VirtualAdversarialMethod, self).__init__(model, back, sess)

        self.feedable_kwargs = {'eps': tf.float32, 'xi': tf.float32,
                                'clip_min': tf.float32,
                                'clip_max': tf.float32}
        self.structural_kwargs = ['num_iterations']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float ) the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        return vatm(self.model, x, self.model.get_logits(x), eps=self.eps,
                    num_iterations=self.num_iterations, xi=self.xi,
                    clip_min=self.clip_min, clip_max=self.clip_max)

    def parse_params(self, eps=2.0, num_iterations=1, xi=1e-6, clip_min=None,
                     clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float )the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        self.eps = eps
        self.num_iterations = num_iterations
        self.xi = xi
        self.clip_min = clip_min
        self.clip_max = clip_max
        return True


def vatm(model, x, logits, eps, back='tf', num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None):
    """
    A wrapper for the perturbation methods used for virtual adversarial
    training : https://arxiv.org/abs/1507.00677
    It calls the right function, depending on the
    user's backend.
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    from attacks_tf import vatm as vatm_tf
    return vatm_tf(model, x, logits, eps, num_iterations=num_iterations, xi=xi, clip_min=clip_min, clip_max=clip_max)
