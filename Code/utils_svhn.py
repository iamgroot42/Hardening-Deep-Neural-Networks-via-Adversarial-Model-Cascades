import common

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def placeholder_shapes():
	return (None, 3, 32, 32), (None, 10)


def data_svhn():
	img_rows = 32
	img_cols = 32
	nb_classes = 10
	# the data, shuffled and split between train and test sets
	X_train, y_train = np.load("SVHNx_tr.npy"), np.load("SVHNy_tr.npy")
	X_test, y_test = np.load("SVHNx_te.npy"), np.load("SVHNy_te.npy")
	X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return X_train, Y_train, X_test, Y_test


def augmented_data(X):
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=False,  # randomly flip images
		vertical_flip=True,  # randomly flip images
		data_format="channels_first") # (channel, row, col) format per image
	datagen.fit(X)
	return datagen
