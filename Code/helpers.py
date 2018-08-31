import common

import tensorflow as tf
import numpy as np
import copy, math, sys
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, DeepFool, ElasticNetMethod, SaliencyMapMethod, MadryEtAl, MomentumIterativeMethod, VirtualAdversarialMethod
from keras import backend as K
from tqdm import tqdm

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
		attack_params['nb_iter'] = 3
	elif attack_name == "fgsm":
		attack_object = FastGradientMethod(model, sess=session)
		if harden:
			if dataset == "mnist":
				attack_params['eps'] = 0.3
			else:
				attack_params['eps'] = 0.1
	elif attack_name == "elastic":
		attack_object = ElasticNetMethod(model, sess=session)
		attack_params['beta'] = 1e-2
		attack_params['binary_search_steps'] = 1
		attack_params['max_iterations'] = 5
		attack_params['initial_const'] = 1e-1
		attack_params['learning_rate'] = 1e-1
		if dataset == "mnist":
			attack_params['learning_rate'] = 1e-1
			attack_params['initial_const'] = 1e-3
			attack_params['binary_search_steps'] = 4
			attack_params['max_iterations'] = 8
	elif attack_name == "virtual":
		attack_object = VirtualAdversarialMethod(model, sess=session)
		attack_params['xi'] = 1e-6
		attack_params['num_iterations'] = 1
		attack_params['eps'] = 2.0
		if dataset == "mnist":
			attack_params['num_iterations'] = 6
			attack_params['xi'] = 1e0
			attack_params['eps'] = 5.0
	elif attack_name == "madry":
		attack_object = MadryEtAl(model, sess=session)
		attack_params['nb_iter'] = 10
		if harden:
			if dataset == "mnist":
				attack_params['eps'] = 0.1
			else:
				attack_params['eps'] = 0.03
	elif attack_name == "jsma":
		attack_object = SaliencyMapMethod(model, sess=session)
		attack_params['gamma'] = 0.1
		attack_params['theta'] = 1.0
	elif attack_name == "carlini":
		if dataset == "cifar10":
			attack_params["confidence"] = 0.0
			attack_params["max_iterations"] = 100
			attack_params["binary_search_steps"] = 20
			attack_params["abort_early"] = False
			attack_params["initial_const"] = 1e-4
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
	for i in tqdm(range(0, attack_X.shape[0], batch_size)):
		mini_batch = attack_X[i: i + batch_size,:]
		if mini_batch.shape[0] == 0:
			break
		attack_params['batch_size'] = mini_batch.shape[0]
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
			scheduler, batch_size, attacks=None, early_stop=None, lr_plateau=None):

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

	# Variables for early stopping
	best_loss = 1e6
	best_acc = 0
	wait      = 0
	if early_stop:
		min_delta, patience = early_stop

	# Variables for lerning rate decay based on le_plateau
	if lr_plateau and scheduler:
		print("Cannot have scheduler and lr_plateau together. Pick one of them and use")
		return
	lrp_wait      = 0
	lrp_best_loss = 1e6
	lrp_best_acc = 0
	if lr_plateau:
		min_lr, factor, lrp_patience, lrp_min_delta = lr_plateau

	# Iterate over epochs
	for j in range(epochs):
		train_loss, val_loss = 0, 0
		train_acc, val_acc = 0, 0
		iterator = dataGen.flow(X_train, Y_train, batch_size=batch_size)
		nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
		assert nb_batches * batch_size >= len(X_train)

		# Use scheduler (if provided) to schedule learning rate
		if scheduler:
			K.set_value(model.optimizer.lr, scheduler(j))

		# Iterate over batches
		for batch in range(nb_batches):
			(batchX, batchY), indeces = next(iterator)
			clean_X, clean_Y = X_train[indeces], Y_train[indeces]
			if attacks:
				# If given batch cannot be split for attacks (extreme case, at max 1/epoch), skip batch
				if batchX.shape[0] < len(attacks):
					continue
			# Add attack data if attacks specified
			if attacks:
				additionalX, additionalY = [], []
				attack_indices = np.array_split(np.random.permutation(len(clean_Y)), len(attacks))
				# Add equal amount of data per attack
				for i, (attack, attack_params) in enumerate(attacks):
					attack_params['batch_size'] = clean_X[attack_indices[i]].shape[0]
					adv_data = attack.generate_np(clean_X[attack_indices[i]], **attack_params)
					additionalX.append(adv_data)
					additionalY.append(clean_Y[attack_indices[i]])
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
		# Calculate validation accuracy for adversarial data as well (if attacks provided)
		print(">> Val loss: %f, Val acc: %f"% (val_metrics[0], val_metrics[1]))
		adv_loss, adv_acc = 0, 0
		if attacks:
			for i, (attack, attack_params) in enumerate(attacks):
				attack_params['batch_size'] = X_val.shape[0]
				adv_data = performBatchwiseAttack(X_val, attack, attack_params, batch_size)
				adv_val_metrics = model.evaluate(adv_data, Y_val, batch_size=1024, verbose=0)
				adv_loss += adv_val_metrics[0]
				adv_acc += adv_val_metrics[1]
				print(">> Attack %s: loss: %f, acc: %f"% (attack, adv_val_metrics[0], adv_val_metrics[1]))
			adv_loss /= len(attacks)
			adv_acc /= len(attacks)
			print
		# Early stopping check
		if early_stop:
			current_loss = val_metrics[0]
			current_acc = val_metrics[1]
			if attacks:
				current_loss += adv_loss
				current_acc  += adv_acc
				current_acc  /= 2
			#print(best_loss, current_loss, min_delta, wait)
			print("Best acc: %f, current acc: %f, wait value: %d" % (best_acc, current_acc, wait))
			#if  best_loss - current_loss > min_delta:
			if current_acc - best_acc > min_delta:
				wait = 0
				best_loss = current_loss
				best_acc = current_acc
			else:
				wait += 1
				if wait >= patience:
					# Stop training
					return True

		# LR Pleateau check
		if lr_plateau:
			current_loss = val_metrics[0]
			current_acc  = val_metrics[1]
			if attacks:
				current_loss += adv_loss
				current_acc  += adv_acc
				current_acc  /= 2
			#print(lrp_best_loss, current_loss, lrp_min_delta, lrp_wait)
			print("Best acc: %f, current acc: %f, wait value: %d" % (lrp_best_acc, current_acc, lrp_wait))
			current_lr = float(K.get_value(model.optimizer.lr))
			#if lrp_best_loss - current_loss > lrp_min_delta:
			if current_acc - lrp_best_acc > lrp_min_delta:
				lrp_wait = 0
				lrp_best_loss = current_loss
				lrp_best_acc = current_acc
			else:
				lrp_wait += 1
				if lrp_wait >= lrp_patience:
					if current_lr >= min_lr:
						# Reduce LR
						new_lr = max(current_lr * factor, min_lr)
						K.set_value(model.optimizer.lr, new_lr)
						print("\n Reduced learning rate to %f" % (new_lr))
						lrp_wait = 0
	return True

