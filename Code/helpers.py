import common

import tensorflow as tf
import numpy as np
import copy, math, sys
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, DeepFool, ElasticNetMethod, SaliencyMapMethod, MadryEtAl, MomentumIterativeMethod, VirtualAdversarialMethod
from keras import backend as K


# Return attack object and its appropriate attack parameters for the given attack and dataset
def get_appropriate_attack(dataset, clip_range, attack_name, model, session, harden, attack_type):
	# Check if valid dataset specified
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


# Given an adversarial attack object, perform it batch-wise
def performBatchwiseAttack(attack_X, attack, attack_params, batch_size):
	perturbed_X = np.array([])
	for i in range(0, attack_X.shape[0], batch_size):
		mini_batch = attack_X[i: i + batch_size,:]
		if mini_batch.shape[0] == 0:
			break
		adv_x_mini = attack.generate_np(mini_batch, **attack_params)
		if perturbed_X.shape[0] != 0:
			perturbed_X = np.append(perturbed_X, adv_x_mini, axis=0)
		else:
			perturbed_X = adv_x_mini
	return perturbed_X


# Custom training model (supports adversarial training)
def customTrainModel(model,
			X_train, Y_train,
			X_val, Y_val,
			dataGen, epochs,
			scheduler, batch_size, attacks=None):

	# Helper function to generate adversarial data
	def get_adv_mixed(P, Q):
		additionalX, additionalY = [], []
		attack_indices = np.array_split(np.random.permutation(len(Q)), len(attacks))
		# Add equal amount of data per attack
		for i, (attack, attack_params) in enumerate(attacks):
			adv_data = attack.generate_np(P[attack_indices[i]], **attack_params)
			additionalX.append(adv_data)
			additionalY.append(Q[attack_indices[i]])
		additionalX = np.concatenate(additionalX, axis=0)
		additionalY = np.concatenate(additionalY, axis=0)
		return additionalX, additionalY

	for j in range(epochs):
		train_loss, val_loss = 0, 0
		train_acc, val_acc = 0, 0
		iterator = dataGen.flow(X_train, Y_train, batch_size=batch_size)
		nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
		assert nb_batches * batch_size >= len(X_train)
		if scheduler:
			K.set_value(model.optimizer.lr, scheduler(j))
		for batch in range(nb_batches):
			plainX, plainY = next(iterator)
			batchX, batchY = plainX, plainY
			# Add attack data if attacks specified
			if attacks:
				additionalX, additionalY = [], []
				attack_indices = np.array_split(np.random.permutation(len(plainY)), len(attacks))
				# Add equal amount of data per attack
				for i, (attack, attack_params) in enumerate(attacks):
					adv_data = attack.generate_np(plainX[attack_indices[i]], **attack_params)
					additionalX.append(adv_data)
					additionalY.append(plainY[attack_indices[i]])
				additionalX = np.concatenate(additionalX, axis=0)
				additionalY = np.concatenate(additionalY, axis=0)
				#additionalX, additionalY = get_adv_mixed(plainX, plainY)
				batchX = np.concatenate((batchX, additionalX), axis=0)
				batchY = np.concatenate((batchY, additionalY), axis=0)
			train_metrics = model.train_on_batch(batchX, batchY)
			train_loss += train_metrics[0]
			train_acc += train_metrics[1]
			sys.stdout.write("Epoch %d: %d / %d : Tr loss: %f, Tr acc: %f  \r" % (j+1, batch+1, nb_batches, train_loss/(batch+1), train_acc/(batch+1)))
			sys.stdout.flush()
		val_metrics = model.evaluate(X_val, Y_val, batch_size=1024, verbose=0)
		print
		if attacks:
			attack_indices = np.array_split(np.random.permutation(len(Y_val)), len(attacks))
			adv_val_x, adv_val_y = [], []
			for i, (attack, attack_params) in enumerate(attacks):
				adv_data = performBatchwiseAttack(X_val[attack_indices[i]], attack, attack_params, batch_size)
				adv_val_x.append(adv_data)
				adv_val_y.append(Y_val[attack_indices[i]])
			adv_val_x = np.concatenate(adv_val_x, axis=0)
			adv_val_y = np.concatenate(adv_val_y, axis=0)
			adv_val_metrics = model.evaluate(adv_val_x, adv_val_y, batch_size=1024, verbose=0)
			print(">> Val loss: %f, Val acc: %f, Adv loss: %f, Adv acc: %f"% (val_metrics[0], val_metrics[1], adv_val_metrics[0], adv_val_metrics[1]))
		else:
			print(">> Val loss: %f, Val acc: %f"% (val_metrics[0], val_metrics[1]))
	return True
