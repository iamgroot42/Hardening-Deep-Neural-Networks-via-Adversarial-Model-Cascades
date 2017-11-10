import common
import numpy as np

from keras.models import load_model
from tensorflow.python.platform import app

from keras.models import Sequential
from keras.layers import Activation

import tensorflow as tf
import keras
from tensorflow.python.platform import flags
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Models import cnn

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_float('learning_rate', 1 ,'Learning rate for classifier')
flags.DEFINE_string('save_here', 'saved_model', 'Path where model is to be saved')
flags.DEFINE_string('teacher_model', 'saved_model', 'Path where teacher model (blackbox) is stored')
flags.DEFINE_string('unlabelled_data', 'X.npy', 'Unlabelled data used by student to get labels from teacher')
flags.DEFINE_boolean('use_distillation', False, 'Whether model is based on distillation or a normal proxy')
flags.DEFINE_float('label_smooth', 0, 'Amount of label smoothening to be applied')


def main(argv=None):
	"""
	Train a network using defensive distillation (if distillation specified; use direct predictions otherwise)

	Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
	Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
	IEEE S&P, 2016.
	"""

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	nb_classes = 10
	shape = (3, 32, 32)
	if FLAGS.dataset == "mnist":
		shape = (1, 28, 28)
	elif FLAGS.dataset not in ["cifar10", "svhn"]:
		print("Invalid dataset!")
		exit()

	model = None
	if FLAGS.use_distillation:
		model = cnn.proxy(shape, nb_classes, FLAGS.learning_rate, True)
	else:
		model = cnn.proxy(shape, nb_classes, FLAGS.learning_rate)

	# load teacher model, unlabelled data
	X_train = np.load(FLAGS.unlabelled_data)
	teacher = load_model(FLAGS.teacher_model)

	# Convert to one-hot if not distillation
	Y_train = teacher.predict(X_train)
	if not FLAGS.use_distillation:
		temp = np.zeros(Y_train.shape)
		temp[np.arange(Y_train.shape[0]), np.argmax(Y_train, axis=1)] = 1
		Y_train = temp
		if FLAGS.label_smooth > 0:
	                Y_train = Y_train.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)


	# Early stopping and dynamic lr
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.001, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1)

	# train the student model
	model.fit(X_train, Y_train,
		batch_size=16,
		validation_split=0.2,
		epochs=FLAGS.nb_epochs,
		callbacks=[reduce_lr, early_stop])

	# save student model (add softmax if distillation model)
	if FLAGS.use_distillation:
		model.add(Activation('softmax'))
	model.save(FLAGS.save_here)


if __name__ == '__main__':
	app.run()
