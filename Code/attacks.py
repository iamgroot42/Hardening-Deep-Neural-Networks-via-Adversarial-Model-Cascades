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
	def get_or_guess_labels(self, x, kwargs):
	        """
        	Get the label to use in generating an adversarial example for x.
	        The kwargs are fed directly from the kwargs of the attack.
	        If 'y' is in kwargs, then assume it's an untargeted attack and
	        use that as the label.
	        If 'y_target' is in kwargs, then assume it's a targeted attack and
	        use that as the label.
	        Otherwise, use the model's prediction as the label and perform an
	        untargeted attack.
	        """

	        if 'y' in kwargs and 'y_target' in kwargs:
        	    raise ValueError("Can not set both 'y' and 'y_target'.")
	        elif 'y' in kwargs:
        	    labels = kwargs['y']
	        elif 'y_target' in kwargs:
	            labels = kwargs['y_target']
	        else:
	            preds = self.model.get_probs(x)
	            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
	            original_predictions = tf.to_float(tf.equal(preds,
	                                                        preds_max))
	            labels = tf.stop_gradient(original_predictions)
	        if isinstance(labels, np.ndarray):
	            nb_classes = labels.shape[1]
	        else:
	            nb_classes = labels.get_shape().as_list()[1]
	        return labels, nb_classes

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
			return attacks_tf.deepfool_batch(self.sess, x, preds, logits, grads, x_val,
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


class ElasticNetMethod(Attack):
    def __init__(self, model, back='tf', sess=None):
        super(ElasticNetMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['beta', 'batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param beta: Trades off L2 distortion with L1 distortion: higher
                     produces examples with lower L1 distortion, at the
                     cost of higher L2 (and typically Linf) distortion
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
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
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        self.parse_params(**kwargs)

        from attacks_tf import ElasticNetMethod as EAD
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = EAD(self.sess, self.model, self.beta,
                     self.batch_size, self.confidence,
                     'y_target' in kwargs, self.learning_rate,
                     self.binary_search_steps, self.max_iterations,
                     self.abort_early, self.initial_const,
                     self.clip_min, self.clip_max,
                     nb_classes, x.get_shape().as_list()[1:])

        def ead_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(ead_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None,
                     nb_classes=None, beta=1e-3,
                     batch_size=9, confidence=0,
                     learning_rate=1e-2,
                     binary_search_steps=9, max_iterations=1000,
                     abort_early=False, initial_const=1e-3,
                     clip_min=0, clip_max=1):

        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.beta = beta
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max


class MadryEtAl(Attack):
    """
    The Projected Gradient Descent Attack (Madry et al. 2016).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """
    def __init__(self, model, back='tf', sess=None):
        super(MadryEtAl, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']


    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
            error_string = ("ProjectedGradientDescentMethod is"
                            " not implemented in Theano")
            raise NotImplementedError(error_string)

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        from utils_tf import tf_model_loss, clip_eta

        adv_x = x + eta
        preds = self.model.get_probs(adv_x)
        loss = tf_model_loss(y, preds)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return x, eta

    def attack(self, x, **kwargs):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.
        :param x: A tensor with the input image.
        """
        from utils_tf import clip_eta

        eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
        eta = clip_eta(eta, self.ord, self.eps)

        if self.y is not None:
            y = self.y
        else:
            preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            y = tf.to_float(tf.equal(preds, preds_max))
            y = y / tf.reduce_sum(y, 1, keep_dims=True)
        y = tf.stop_gradient(y)

        for i in range(self.nb_iter):
            x, eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x
