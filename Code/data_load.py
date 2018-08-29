import common

from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10, mnist


class Data:
	def __init__(self, dataset, extra_X, extra_Y):
		self.dataset = dataset
		self.extra_X = None
		self.extra_Y = None
		self.threshold = 3000 # Total Adversarial examples
		self.clip_min = 0.
		self.clip_max = 1.
		if extra_X is not None:
			assert(extra_Y is not None)
			self.extra_X = extra_X
			self.extra_Y = extra_Y

	def make_val_data(self):
		# Class balanced: 30% validation, 70% test
		assert(self.X_test is not None and self.Y_test is not None)
		self.X_test, self.Y_test, self.X_val, self.Y_val = self.data_split(self.X_test, self.Y_test, these_many=int(0.7 * len(self.X_test)))

	def get_placeholder_shape(self):
		data_shape = (None,) + self.X_train.shape[1:]
		label_shape = (None,) + (self.Y_train.shape[1],)
		return data_shape, label_shape

	def get_range(self):
		return (self.clip_min, self.clip_max)

	def data_generator(self):
		datagen = ImageDataGenerator()
		return datagen

	def validation_split(self, X, Y, validation_split=0.1):
		num_points = len(X)
		validation_indices = np.random.choice(num_points, int(num_points * validation_split))
		train_indices = list(set(range(num_points)) - set(validation_indices))
		X_train, y_train = X[train_indices], Y[train_indices]
		X_val, y_val = X[validation_indices], Y[validation_indices]
		return X_train, y_train, X_val, y_val

	def data_split(self, X, Y, these_many):
		nb_classes = Y.shape[1]
		these_many /= nb_classes
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
		n_points = min([len(distr[i]) for i in distr.keys()])
		for key in distr.keys():
			st = np.random.choice(distr[key], n_points, replace=False)
			bm = st[:these_many]
			pm = st[these_many:]
			X_bm_ret.append(X[bm])
			Y_bm_ret.append(Y[bm])
			X_pm_ret.append(X[pm])
			Y_pm_ret.append(Y[pm])
		X_bm_ret = np.concatenate(X_bm_ret)
		Y_bm_ret = np.concatenate(Y_bm_ret)
		X_pm_ret = np.concatenate(X_pm_ret)
		Y_pm_ret = np.concatenate(Y_pm_ret)
		return X_bm_ret, Y_bm_ret, X_pm_ret, Y_pm_ret

	def experimental_split(self):
		# Extract training and test data for blackbox from original training data
		self.blackbox_Xtrain, self.blackbox_Ytrain = self.X_train, self.Y_train

		# Add additonal data if present:
		if self.extra_X is not None:
			self.blackbox_Xtrain = np.concatenate((self.blackbox_Xtrain, self.extra_X))
			self.blackbox_Ytrain = np.concatenate((self.blackbox_Ytrain, self.extra_Y))

		self.attack_X, self.attack_Y, self.blackbox_Xtest, self.blackbox_Ytest = self.data_split(self.X_test, self.Y_test, self.threshold)

	def get_blackbox_data(self):
		return (self.blackbox_Xtrain, self.blackbox_Ytrain), (self.blackbox_Xtest, self.blackbox_Ytest)

	def get_attack_data(self):
		return (self.attack_X, self.attack_Y)

	def get_validation_data(self):
		return (self.X_val, self.Y_val)


class SVHN(Data, object):
	def __init__(self, extra_X=None, extra_Y=None):
		super(SVHN, self).__init__("svhn", extra_X, extra_Y)
		# the data, shuffled and split between train and test sets
		self.X_train, self.Y_train = np.load("../Code/SVHN/SVHNx_tr.npy"), np.load("../Code/SVHN/SVHNy_tr.npy")
		self.X_test, self.Y_test = np.load("../Code/SVHN/SVHNx_te.npy"), np.load("../Code/SVHN/SVHNy_te.npy")
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		# convert class vectors to binary class matrices
		self.Y_train = np_utils.to_categorical(self.Y_train - 1, 10)
		self.Y_test = np_utils.to_categorical(self.Y_test - 1, 10)
		self.X_train /= 255.0
		self.X_test /= 255.0
		super(SVHN, self).make_val_data()
		super(SVHN, self).experimental_split()


	def data_generator(self):
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
			data_format="channels_last") # (row, col, channel) format per image
		return datagen


class CIFAR10(Data, object):
	def __init__(self, extra_X=None, extra_Y=None):
		super(CIFAR10, self).__init__("cifar10", extra_X, extra_Y)
		# the data, shuffled and split between train and test sets
		(self.X_train, self.Y_train), (self.X_test, self.Y_test) = cifar10.load_data()
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_test /= 255
		self.X_train /= 255
		# convert class vectors to binary class matrices
		self.Y_train = np_utils.to_categorical(self.Y_train, 10)
		self.Y_test = np_utils.to_categorical(self.Y_test, 10)
		super(CIFAR10, self).make_val_data()
		super(CIFAR10, self).experimental_split()

	def data_generator(self):
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
			data_format="channels_last") # (row, col, channel) format per image
		return datagen


class MNIST(Data, object):
	def __init__(self, extra_X=None, extra_Y=None):
		super(MNIST, self).__init__("mnist", extra_X, extra_Y)
		# the data, shuffled and split between train and test sets
		(self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()
		self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train /= 255
		self.X_test /= 255
		# convert class vectors to binary class matrices
		self.Y_train = np_utils.to_categorical(self.Y_train, 10)
		self.Y_test = np_utils.to_categorical(self.Y_test, 10)
		super(MNIST, self).make_val_data()
		super(MNIST, self).experimental_split()


def get_appropriate_data(dataset):
	dataset_mapping = {
		"mnist": MNIST,
		"cifar10": CIFAR10,
		"svhn": SVHN
	}
	if dataset.lower() in dataset_mapping:
		return dataset_mapping[dataset.lower()]
	return None


def get_proxy_data(dataset):
	X = None
	path = {
		"mnist": "MNIST/data.npy",
		"cifar10": "CIFAR10new/data.npy",
		"svhn": "SVHN/data.npy"
	}
	if dataset.lower() in path:
		data = np.load("../Code/SVHN/UnlabelledData/" + path[dataset.lower()])
		data = data.transpose((0, 2, 3, 1))
		return data
	return None

