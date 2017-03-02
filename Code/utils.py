import common

import os
import keras


def save_model(model, filepath, weights_only=False):
	if weights_only:
		# Dump model weights
		model.save_weights(filepath)
		print("Model weights were saved to: " + filepath)
	else:
		# Dump model architecture and weights
		model.save(filepath)
		print("Model was saved to: " + filepath)


def load_model(filepath, weights_only=False, model=None):
	if weights_only:
		assert model is not None
	assert os.path.exists(filepath)
	if weights_only:
		result = model.load_weights(filepath)
		print(result)
		return model.load_weights(filepath)
	else:
		return keras.models.load_model(filepath)


def batch_indices(batch_nb, data_length, batch_size):
	start = int(batch_nb * batch_size)
	end = int((batch_nb + 1) * batch_size)
	# When there are not enough inputs left, we reuse some to complete the batch
	if end > data_length:
		shift = end - data_length
		start -= shift
		end -= shift
	return start, end


def other_classes(nb_classes, class_ind):
	other_classes_list = list(xrange(nb_classes))
	other_classes_list.remove(class_ind)
	return other_classes_list
