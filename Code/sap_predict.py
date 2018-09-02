import common
import argparse
import numpy as np
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Input

from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
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
		# First submodel
		self.sub_models[0].layers[-1].set_weights(weights_model.layers[0].get_weights())
		# Second submodel
		self.sub_models[1].layers[-1].set_weights(weights_model.layers[2].get_weights())
		# Third submodel
		self.sub_models[2].layers[-1].set_weights(weights_model.layers[5].get_weights())
		# Fourth submodel
		self.sub_models[3].layers[-1].set_weights(weights_model.layers[6].get_weights())
		# Fifth submodel
		self.sub_models[4].layers[-1].set_weights(weights_model.layers[7].get_weights())
		print("Set weights!")

	def forward(self, next_input):
		for sub_model in self.sub_models[:-1]:
			output = sub_model.predict(next_input)
			next_input = stochastic_activation(output)
		prediction_output = self.sub_models[-1].predict(next_input)
		return prediction_output

if __name__ == "__main__":
	raw_model = LenetSAP(True)
	model = load_model(args.model_path)

	raw_model.load_from_weights(model)

	try:
		X, Y = np.load(args.data_x), np.load(args.data_y)
	except:
		dataObject = data_load.get_appropriate_data(args.dataset)()
		_, (X, Y) = dataObject.get_blackbox_data()

	acc = 0.
	for j in tqdm(range(X.shape[0])):
		prediciton = raw_model.forward(np.expand_dims(X[j], axis=0))[0]
		acc += 1. * (np.argmax(prediciton) == np.argmax(Y[j]))
	acc /= len(Y)

	print("Prediction accuracy on given data with SAP : %f" % (acc))
