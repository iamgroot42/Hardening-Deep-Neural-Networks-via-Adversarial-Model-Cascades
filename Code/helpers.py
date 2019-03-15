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
	if attack_name == "momentum":
		attack_object = MomentumIterativeMethod(model, sess=session)
		attack_params['eps'], attack_params['eps_iter'], attack_params['nb_iter'] = 0.3, 0.06, 3
	elif attack_name == "fgsm":
		attack_object = FastGradientMethod(model, sess=session)
		if dataset == "mnist":
			attack_params['eps'] = 0.3
			if attack_type == "black":
				attack_params['eps'] = 0.3
		else:
			attack_params['eps'] = 0.1
	elif attack_name == "elastic":
		attack_object = ElasticNetMethod(model, sess=session)
		attack_params['binary_search_steps'], attack_params['max_iterations'], attack_params['beta'] = 1, 5, 1e-2
		attack_params['initial_const'], attack_params['learning_rate'] = 1e-1, 1e-1
		if dataset == "svhn":
			attack_params['initial_const'], attack_params['learning_rate'] = 3e-1, 2e-1
		if attack_type == "black":
			attack_params['max_iterations'], attack_params['binary_search_steps'] = 8, 2
		if dataset == "mnist":
			attack_params['learning_rate'], attack_params['initial_const'] = 1e-1, 1e-3
			attack_params['binary_search_steps'], attack_params['max_iterations'] = 4, 8
			if attack_type == "black":
				attack_params["max_iterations"], attack_params['binary_search_steps'] = 12, 5
	elif attack_name == "virtual":
		attack_object = VirtualAdversarialMethod(model, sess=session)
		attack_params['xi'] = 1e-6
		attack_params['num_iterations'], attack_params['eps'] = 1, 2.0
		if attack_type == "black":
			attack_params['num_iterations'] = 3
			attack_params['xi'], attack_params['eps'] = 1e-4, 3.0
		if dataset == "mnist":
			attack_params['num_iterations'] = 6
			attack_params['xi'], attack_params['eps'] = 1e0, 5.0
			if attack_type == "black":
				attack_params['num_iterations'], attack_params['eps'] = 10, 8.0
	elif attack_name == "madry":
		attack_object = MadryEtAl(model, sess=session)
		attack_params['nb_iter'], attack_params['eps'] = 5, 0.1
		if dataset == "mnist":
			attack_params['eps'], attack_params['nb_iter'] = 0.3, 15
			if attack_type == "black":
				attack_params['nb_iter'] = 20
	elif attack_name == "jsma":
		attack_object = SaliencyMapMethod(model, sess=session)
		attack_params['gamma'], attack_params['theta'] = 0.1, 1.0
	elif attack_name == "carlini":
		if dataset == "cifar10":
			attack_params["confidence"], attack_params["max_iterations"] = 0.0, 100
			attack_params["binary_search_steps"], attack_params["abort_early"] = 20, False
			attack_params["initial_const"] = 1e-4
		attack_object = CarliniWagnerL2(model, sess=session)
	else:
		raise ValueError('Mentioned attack not implemented')
	print(attack_name, ":", attack_params)
	return attack_object, attack_params

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

def customTrainModel(model, X_train, Y_train, X_val, Y_val,
			dataGen, epochs, scheduler, batch_size, attacks=None, early_stop=None, lr_plateau=None):
	def get_adv_mixed(P, Q):
		additionalX, additionalY = [], []
		attack_indices = np.array_split(np.random.permutation(len(Q)), len(attacks))
		for i, (attack, attack_params) in enumerate(attacks):
			adv_data = attack.generate_np(P[attack_indices[i]], **attack_params)
			additionalX.append(adv_data)
			additionalY.append(Q[attack_indices[i]])
		additionalX = np.concatenate(additionalX, axis=0)
		additionalY = np.concatenate(additionalY, axis=0)
		return additionalX, additionalY
	best_loss = 1e6
	best_acc, wait = 0, 0
	if early_stop:
		min_delta, patience = early_stop
	if lr_plateau and scheduler:
		print("Cannot have scheduler and lr_plateau together. Pick one of them and use")
		return
	lrp_wait, lrp_best_acc = 0, 0
	lrp_best_loss = 1e6
	if lr_plateau:
		min_lr, factor, lrp_patience, lrp_min_delta = lr_plateau
	for j in range(epochs):
		train_loss, val_loss = 0, 0
		train_acc, val_acc = 0, 0
		iterator = dataGen.flow(X_train, Y_train, batch_size=batch_size)
		nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
		assert nb_batches * batch_size >= len(X_train)
		if scheduler:
			K.set_value(model.optimizer.lr, scheduler(j))
		for batch in range(nb_batches):
			(batchX, batchY), indeces = next(iterator)
			clean_X, clean_Y = X_train[indeces], Y_train[indeces]
			if attacks:
				if batchX.shape[0] < len(attacks):
					continue
			if attacks:
				additionalX, additionalY = [], []
				permutation = np.random.permutation(len(clean_Y))
				attack_indices = np.array_split(permutation, len(attacks))
				if len(attacks) > 1:
					current_attack = permutation[int(len(clean_Y) * 0.2):]
					old_attacks = permutation[:int(len(clean_Y) * 0.2)]
					attack_indices = np.array_split(old_attacks, len(attacks) - 1)
					attack_indices.append(current_attack)
				for i, (attack, attack_params) in enumerate(attacks):
					attack_params['batch_size'] = clean_X[attack_indices[i]].shape[0]
					adv_data = attack.generate_np(clean_X[attack_indices[i]], **attack_params)
					additionalX.append(adv_data)
					additionalY.append(clean_Y[attack_indices[i]])
				additionalX = np.concatenate(additionalX, axis=0)
				additionalY = np.concatenate(additionalY, axis=0)
				batchX = np.concatenate((batchX, additionalX), axis=0)
				batchY = np.concatenate((batchY, additionalY), axis=0)
			train_metrics = model.train_on_batch(batchX, batchY)
			train_loss += train_metrics[0]
			train_acc += train_metrics[1]
			sys.stdout.write("Epoch %d: %d / %d : Tr loss: %f, Tr acc: %f  \r" % (j+1, batch+1, nb_batches, train_loss/(batch+1), train_acc/(batch+1)))
			sys.stdout.flush()
		val_metrics = model.evaluate(X_val, Y_val, batch_size=1024, verbose=0)
		print
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
		if early_stop:
			current_loss = val_metrics[0]
			current_acc = val_metrics[1]
			if attacks:
				current_loss += adv_loss
				current_acc  += adv_acc
				current_acc  /= 2
			print("Best acc: %f, current acc: %f, wait value: %d" % (best_acc, current_acc, wait))
			if current_acc - best_acc > min_delta:
				best_loss, best_acc, wait = current_loss, current_acc, 0
			else:
				wait += 1
				if wait >= patience:
					return True
		if lr_plateau:
			current_loss = val_metrics[0]
			current_acc  = val_metrics[1]
			if attacks:
				current_loss += adv_loss
				current_acc  += adv_acc
				current_acc  /= 2
			print("Best acc: %f, current acc: %f, wait value: %d" % (lrp_best_acc, current_acc, lrp_wait))
			current_lr = float(K.get_value(model.optimizer.lr))
			if current_acc - lrp_best_acc > lrp_min_delta:
				lrp_best_loss, lrp_best_acc, lrp_wait = current_loss, current_acc, 0
			else:
				lrp_wait += 1
				if lrp_wait >= lrp_patience:
					if current_lr >= min_lr:
						new_lr = max(current_lr * factor, min_lr)
						K.set_value(model.optimizer.lr, new_lr)
						print("\n Reduced learning rate to %f" % (new_lr))
						lrp_wait = 0
	return True