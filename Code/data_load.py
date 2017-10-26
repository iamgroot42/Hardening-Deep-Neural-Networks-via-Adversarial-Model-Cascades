import common

from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10, mnist

# Set seed for reproducability
np.random.seed(42)

class Data:
	def __init__(self, dataset, extra_X, extra_Y):
		self.dataset = dataset
		self.extra_X = None
		self.extra_Y = None
		if extra_X != None:
			assert(extra_Y != None)
			self.extra_X = extra_X
			self.extra_Y = extra_Y

	def placeholder_shape(self):
		data_shape = (None,) + self.data.X_train.shape[1:]
		label_shape = (None,) + (self.data.Y_train.shape[1],)
		return data_shape, label_shape

	def date_generator(self):
		datagen = ImageDataGenerator()
		return datagen

	def validation_split(self, X, Y, validation_split=0.2):
		num_points = len(X)
		validation_indices = np.random.choice(num_points, int(num_points * validation_split))
		train_indices = list(set(range(num_points)) - set(validation_indices))
		X_train, y_train = X[train_indices], y[train_indices]
		X_val, y_val = X[validation_indices], y[validation_indices]
		return X_train, y_train, X_val, y_val

	def data_split(self, X, Y, pool_split=0.8):
		nb_classes = Y.shape[1]
		distr = {}
		for i in range(nb_classes):
			distr[i] = []
		if Y.shape[1] == nb_classes:
			for i in range(len(Y)):
				distr[np.argmax(Y[i])].append(i)
		else:
			for i in range(len(Y)):
				distr[Y[i][0]].append(i)
		X_bm_ret = []
		Y_bm_ret = []
		X_pm_ret = []
		Y_pm_ret = []
		# Calculate minimum number of points per class
		n_points = min([distr[i] for i in distr.keys()])
		for key in distr.keys():
			st = np.random.choice(distr[key], n_points, replace=False)
			bm = st[:int(len(st)*pool_split)]
			pm = st[int(len(st)*pool_split):]
			X_bm_ret.append(X[bm])
			Y_bm_ret.append(Y[bm])
			X_pm_ret.append(X[pm])
			Y_pm_ret.append(Y[pm])
		X_bm_ret = np.concatenate(X_bm_ret)
		Y_bm_ret = np.concatenate(Y_bm_ret)
		X_pm_ret = np.concatenate(X_pm_ret)
		Y_pm_ret = np.concatenate(Y_pm_ret)
		return X_train_bm_ret, Y_train_bm_ret, X_train_pm_ret, Y_train_pm_ret

	def experimental_split(self):
		# Add additonal data if present:
		if self.extra_X:
			self.X_train = np.concatenate((self.X_train, self.extra_X))
			self.Y_train = np.concatenate((self.Y_train, self.extra_Y))
		# Extract training and test data for blackbox from original training data
		(self.blackbox_Xtrain, self.blackbox_Ytrain), (self.blackbox_Xtest, self.blackbox_Ytest) = self.data_split(self.X_train, self.Y_train)
		# Split test data into data for attacking and data used by blackbox for self-proxy hardening
		(self.attack_X, self.attack_Y), (self.harden_X, self.harden_Y) = self.data_split(self.X_test, self.Y_test)

	def get_blackbox_data(self):
		return (self.blackbox_Xtrain, self.blackbox_Ytrain), (self.blackbox_Xtest, self.blackbox_Ytest)

	def get_attack_data(self):
		return (self.attack_X, self.attack_Y)

	def get_hardening_data(self):
		return (self.harden_X, self.harden_Y)


class SVHN(Data):
	def __init__(self, extra_X=None, extra_Y=None):
		super(SVHN, self).__init__("svhn", extra_X, extra_Y)
		# the data, shuffled and split between train and test sets
		self.X_train, self.Y_train = np.load("../Code/SVHN/SVHNx_tr.npy"), np.load("../Code/SVHN/SVHNy_tr.npy")
		self.X_test, self.Y_test = np.load("../Code/SVHN/SVHNx_te.npy"), np.load("../Code/SVHN/SVHNy_te.npy")
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train /= 255.0
		self.X_test /= 255.0
		super(SVHN, self).experimental_split()


	def date_generator(self):
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=False,  # randomly flip images
			vertical_flip=False,  # randomly flip images
			data_format="channels_first") # (channel, row, col) format per image
		return datagen


class CIFAR10(Data):
	def __init__(self, extra_X=None, extra_Y=None):
		super(CIFAR10, self).__init__("cifar10", extra_X, extra_Y)
		# the data, shuffled and split between train and test sets
		(self.X_train, self.Y_train), (self.X_test, self.Y_test) = cifar10.load_data()
		self.X_train = self.X_train.reshape(self.X_train.shape[0], 3, 32, 32)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], 3, 32, 32)
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train /= 255
		self.X_test /= 255
		# convert class vectors to binary class matrices
		self.Y_train = np_utils.to_categorical(self.Y_train, 10)
		self.Y_test = np_utils.to_categorical(self.Y_test, 10)
		super(CIFAR10, self).experimental_split()

	def date_generator(self):
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=True,  # randomly flip images
			data_format="channels_first") # (channel, row, col) format per image
		return datagen


class MNIST(Data):
	def __init__(self, extra_X=None, extra_Y=None):
		super(MNIST, self).__init__("mnist", extra_X, extra_Y)
		# the data, shuffled and split between train and test sets
		(self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()
		self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 28, 28)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 28, 28)
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train /= 255
		self.X_test /= 255
		# convert class vectors to binary class matrices
		self.Y_train = np_utils.to_categorical(self.Y_train, 10)
		self.Y_test = np_utils.to_categorical(self.Y_test, 10)
		super(MNIST, self).experimental_split()


def get_appropriate_data(dataset):
	dataset_mapping = {
		"mnist": MNIST,
		"cifar10": CIFAR10,
		"svhn": SVHN
	}
	if dataset.lower() not in dataset_mapping:
		return dataset_mapping[dataset.lower()]
	return None
