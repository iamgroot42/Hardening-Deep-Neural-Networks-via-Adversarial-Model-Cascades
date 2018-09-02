# Ugly code but hey..as long as it works!

import common
import argparse
import numpy as np
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Input

from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.regularizers import l2

from tqdm import tqdm
import data_load

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-dy','--data_x', type=str, default="", metavar='STRING',
				help='path to images to run inference on')
parser.add_argument('-dx','--data_y', type=str, default="", metavar='STRING',
				help='path to labels to run inference on')
parser.add_argument('-m','--model_path', type=str, default="", metavar='STRING',
				 help='path where trained model is stored')
parser.add_argument('-f','--fraction', type=float, default=1.0, metavar='FLOAT',
				 help='fraction to be used while sampling')
parser.add_argument('-s','--num_samples', type=int, default=100, metavar='NUMBER',
				help='number of times activations should be sampled')
parser.add_argument('-d','--dataset', type=str, default='mnist', metavar='STRING',
				help='dataset on which model is trained')

args = parser.parse_args()


def stochastic_activation(x):
	original_shape = x.shape
	x_flat = x.flatten()
	abs_x = np.abs(x_flat)
	probabilities = abs_x / np.sum(abs_x)
	sampled_indices = set([])
	for i in range(args.num_samples):
		sampled_indices = sampled_indices.union(set(np.random.choice(range(x_flat.shape[0]),
				size=int(args.fraction * x_flat.shape[0]), replace=True, p=probabilities)))
	set_zero_indices = list(set(range(x_flat.shape[0])).difference(set(sampled_indices)))
	output = np.copy(x_flat)
	sampled_indices = list(sampled_indices)
	output[sampled_indices] *= 1. / (1. - np.power(1. - probabilities[sampled_indices], args.num_samples))
	output[set_zero_indices] = 0.
	return output.reshape(x.shape)

class LenetSAP:
	def __init__(self, is_mnist):
		input_shape=(32,32,3)

		if is_mnist:
			input_shape=(28,28,1)

		self.sub_models = []
		# First submodel
		img = Input(input_shape)
		first_act = Conv2D(6, (5, 5), padding='valid', activation = 'relu')(img)
		self.sub_models.append(Model(inputs=img, outputs=first_act))

		# Second submodel
		input_shape = (input_shape[0] - 4, input_shape[1] - 4)
		first_act_input = Input((input_shape[0], input_shape[1], 6))
		x = MaxPooling2D((2, 2), strides=(2, 2))(first_act_input)
		second_act = Conv2D(16, (5, 5), padding='valid', activation = 'relu',)(x)
		self.sub_models.append(Model(inputs=first_act_input, outputs=second_act))

		# Third submodel
		input_shape = (input_shape[0]/2 - 4, input_shape[1]/2 - 4)
		third_act_input = Input((input_shape[0], input_shape[1], 16))
		x = MaxPooling2D((2, 2), strides=(2, 2))(third_act_input)
		x = Flatten()(x)
		third_act = Dense(120, activation = 'relu',)(x)
		self.sub_models.append(Model(inputs=third_act_input, outputs=third_act))

		# Fourth submodel
		fourth_act_input = Input((120,))
		fourth_act = Dense(84, activation = 'relu',)(fourth_act_input)
		self.sub_models.append(Model(inputs=fourth_act_input, outputs=fourth_act))

		# Final submodel
		fifth_act_input = Input((84,))
		output = Dense(10, activation = 'softmax',)(fifth_act_input)
		self.sub_models.append(Model(inputs=fifth_act_input, outputs=output))

	def load_from_weights(self, weights_model):
		indices = [0, 2, 5, 6, 7]
		for i, index in enumerate(indices):
			self.sub_models[i].layers[-1].set_weights(weights_model.layers[index].get_weights())
		print("Set weights!")

	def forward(self, next_input):
		for sub_model in self.sub_models[:-1]:
			output = sub_model.predict(next_input)
			next_input = stochastic_activation(output)
		prediction_output = self.sub_models[-1].predict(next_input)
		return prediction_output


hack = False
class Resnet32SAP:
	def __init__(self, is_mnist):
		input_shape=(32,32,3)

		if is_mnist:
			input_shape=(28,28,1)

		self.sub_models = []
		# Submodel 1
		img_input = Input(input_shape)
		x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same')(img_input)
		o1 = Activation('relu')(BatchNormalization()(x))
		self.sub_models.append(Model(inputs=img_input, outputs=o1))

		def residual_block(n_times, in_filters, n_filters, XS, YS, project=None):
			stride = (1,1)
			if project:
				stride=(2,2)
			for _ in range(n_times):
				# Submodel-A
				act_input = Input((XS, YS, in_filters))
				conv = Conv2D(n_filters,kernel_size=(3,3),strides=stride,padding='same')(act_input)
				out_act = Activation('relu')(BatchNormalization()(conv))
				self.sub_models.append(Model(inputs=act_input, outputs=out_act))

				# Submodel-B
				conv_in_shape = (XS, YS, n_filters)
				skip_in_shape = (XS, YS, n_filters)
				if project:
					conv_in_shape = (XS/2, YS/2, n_filters)
					skip_in_shape = (XS, YS, in_filters)
				skip_input = Input(skip_in_shape)
				conv_input = Input(conv_in_shape)
				conv_output = Conv2D(n_filters,kernel_size=(3,3),strides=(1,1),padding='same')(conv_input)
				if project:
					skip_output = Conv2D(n_filters,kernel_size=(1,1),strides=(2,2),padding='same')(skip_input)
				else:
					skip_output = skip_input
				block = add([conv_output, skip_output])
				act = Activation('relu')(BatchNormalization()(block))
				self.sub_models.append(Model(inputs=[conv_input, skip_input], outputs=act))

		residual_block(5, 16, 16, input_shape[0], input_shape[1])
		residual_block(1, 16, 32, input_shape[0], input_shape[1], True)
		residual_block(4, 32, 32, input_shape[0]/2, input_shape[1]/2)
		residual_block(1, 32, 64, input_shape[0]/2, input_shape[1]/2, True)
		residual_block(4, 64, 64, input_shape[0]/4, input_shape[1]/4)

		logits_input = Input((input_shape[0]/4, input_shape[0]/4, 64))
		ga_norm = GlobalAveragePooling2D()(logits_input)
		logits = Dense(10)(ga_norm)
		if hack:
			logits = Dense(10, activation='softmax')(ga_norm)
		output = Activation('softmax')(logits)
		self.sub_models.append(Model(inputs=logits_input, outputs=output))

	def load_from_weights(self, weights_model):
		# First block
		self.sub_models[0].layers[1].set_weights(weights_model.layers[1].get_weights())
		self.sub_models[0].layers[2].set_weights(weights_model.layers[2].get_weights())

		def load_residual_block(base, start, end, offset):
			for i in range(start, end, 2):
				self.sub_models[i + base].layers[1].set_weights(weights_model.layers[offset].get_weights())
				self.sub_models[i + base].layers[2].set_weights(weights_model.layers[offset + 1].get_weights())
				self.sub_models[i + base + 1].layers[1].set_weights(weights_model.layers[offset + 3].get_weights())
				self.sub_models[i + base + 1].layers[4].set_weights(weights_model.layers[offset + 5].get_weights())
				offset += 7
			return offset

		def load_post_block(lhs_point, offset):
			self.sub_models[lhs_point].layers[1].set_weights(weights_model.layers[offset].get_weights())
			self.sub_models[lhs_point].layers[2].set_weights(weights_model.layers[offset + 1].get_weights())
			self.sub_models[lhs_point + 1].layers[2].set_weights(weights_model.layers[offset + 3].get_weights())
			self.sub_models[lhs_point + 1].layers[3].set_weights(weights_model.layers[offset + 4].get_weights())
			self.sub_models[lhs_point + 1].layers[5].set_weights(weights_model.layers[offset + 6].get_weights())
			return offset + 8

		start_point = 4
		start_point = load_residual_block(1, 0, 10, start_point)
		start_point = load_post_block(11, start_point)
		start_point = load_residual_block(13, 0, 8, start_point)
		start_point = load_post_block(21, start_point)
		start_point = load_residual_block(23, 0, 8, start_point)

		# Dense activation layer
		self.sub_models[-1].layers[2].set_weights(weights_model.layers[start_point + 1].get_weights())
		print("Set weights!")

	def forward(self, next_input):
		# Submodel 1
		prev_output = stochastic_activation(self.sub_models[0].predict(next_input))
		for i in range(1, len(self.sub_models) - 1, 2):
			this_output = stochastic_activation(self.sub_models[i].predict(prev_output))
			prev_output = stochastic_activation(self.sub_models[i + 1].predict([this_output, prev_output]))
		prediction_output = self.sub_models[-1].predict(prev_output)
		return prediction_output


if __name__ == "__main__":
	print("== CONSTRUCTING SUBMODELS ==")
	if args.dataset == "mnist":
		raw_model = LenetSAP(True)
	else:
		raw_model = Resnet32SAP(False)

	model = load_model(args.model_path)

	print("== LOADING WEIGHTS INTO SUBMODELS ==")
	raw_model.load_from_weights(model)

	try:
		X, Y = np.load(args.data_x), np.load(args.data_y)
	except:
		dataObject = data_load.get_appropriate_data(args.dataset)()
		_, (X, Y) = dataObject.get_blackbox_data()

	acc = 0.
	X, Y = X[:10], Y[:10]
	for j in tqdm(range(X.shape[0])):
		prediction = raw_model.forward(np.expand_dims(X[j], axis=0))[0]
		acc += 1. * (np.argmax(prediction) == np.argmax(Y[j]))
	acc /= len(Y)

	print("Prediction accuracy on given data with SAP : %f" % (acc))
