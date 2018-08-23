import common

import tensorflow as tf
import numpy as np
import copy
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, DeepFool, ElasticNetMethod, SaliencyMapMethod, MadryEtAl, MomentumIterativeMethod


def get_appropriate_attack(dataset, clip_range, attack_name, model, session, harden, attack_type):
	# CHeck if valid dataset specified
	if dataset not in ["mnist", "svhn", "cifar10"]:
		raise ValueError('Mentioned dataset not implemented')

	attack_object = None
	attack_params = {'clip_min': clip_range[0], 'clip_max': clip_range[1]}

	# Check if valid attack specified, construct object accordingly
	if attack_name == "momentum":
		attack_object = MomentumIterativeMethod(model, sess=session)
		attack_params['eps'] = 0.3
		attack_params['eps_iter'] = 0.06
		attack_params['nb_iter'] = 10
	if attack_name == "fgsm":
		attack_object = FastGradientMethod(model, sess=session)
		if harden:
			if dataset == "mnist":
				attack_params['eps'] = 0.1
			else:
				attack_params['eps'] = 0.03
		else:
			if dataset == "mnist":
				attack_params['eps'] = 0.25
			else:
				attack_params['eps'] = 0.06
	elif attack_name == "elastic":
		attack_object = ElasticNetMethod(model, sess=session)
		attack_params['beta'] = 1e-2
	elif attack_name == "virtual":
		attack_object = VirtualAdversarialMethod(model, sess=session)
		attack_params['xi'] = 1e-6
		attack_params['num_iterations'] = 1
		attack_params['eps'] = 2.0
	elif attack_name == "madry":
		attack_object = MadryEtAl(model, sess=session)
		if harden:
			if dataset == "mnist":
				attack_params['eps'] = 0.1
			else:
				attack_params['eps'] = 0.03
		else:
			if dataset == "mnist":
				attack_params['eps'] = 0.1
			else:
				if attack_type == "white":
					attack_params['eps'] = 0.06
				else:
					attack_params['eps'] = 0.03
	elif attack_name == "jsma":
		attack_object = SaliencyMapMethod(model, sess=session)
		attack_params['gamma'] = 0.1
		attack_params['theta'] = 1.0
	elif attack_name == "c&w":
		attack_object = CarliniWagnerL2(model, sess=session)
	else:
		raise ValueError('Mentioned attack not implemented')

	# Print attack parameters for user's reference
	print(attack_name, ":", attack_params)

	# Return attack object, along with parameters
	return attack_object, attack_params
