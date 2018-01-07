import common

import numpy as np

import keras
from keras.models import load_model

from keras.objectives import categorical_crossentropy
from tensorflow.python.platform import app
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os

import data_load

import tensorflow as tf
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs')
flags.DEFINE_float('sample_ratio', 0.75, 'Percentage of sample to be taken per model for training')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('mode', 'finetune', '(test,finetune)')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_string('model_dir', './', 'path to output directory of models')
flags.DEFINE_string('seed_model', ' ', 'path to seed model')
flags.DEFINE_string('data_x', './', 'path to numpy file of data for prediction')
flags.DEFINE_string('data_y', './', 'path to numpy file of labels for prediction')
flags.DEFINE_float('learning_rate', 0.001 ,'Learning rate for classifier')
flags.DEFINE_string('predict_mode', 'weighted', 'Method for prediction while testing (voting/weighted)')


class Bagging:
	def __init__(self, n_classes, sample_ratio, batch_size, nb_epochs):
		self.n_classes = n_classes
		self.sample_ratio = sample_ratio
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs

	def train(self, X, Y, dataObject, model):
		subset = np.random.choice(len(Y), int(len(Y) * self.sample_ratio))
		x_sub = X[subset]
		y_sub = Y[subset]
		X_tr, y_tr, X_val, y_val = dataObject.validation_split(x_sub, y_sub, 0.2)
		# Early stopping and dynamic lr
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.01, verbose=1)
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=1)
		datagen = dataObject.date_generator()
		datagen.fit(X_tr)
		model.fit_generator(datagen.flow(X_tr, y_tr,
			  			batch_size=self.batch_size),
						steps_per_epoch=X_tr.shape[0] // self.batch_size,
						epochs= self.nb_epochs,
						callbacks=[reduce_lr, early_stop],
						validation_data=(X_val, y_val))
		accuracy = model.evaluate(X_val, y_val, batch_size=self.batch_size)
		print("\nValidation accuracy: " + str(accuracy[1]*100))

	def predict(self, models_dir, predict_on, method='voting'):
		models = []
		for file in os.listdir(models_dir):
			models.append(load_model(os.path.join(models_dir,file)))
		predictions = []
		for model in models:
			predictions.append(model.predict(predict_on))
		if method == 'voting':
			ultimate = [ {i:0 for i in range(self.n_classes)} for j in range(len(predict_on))]
			for prediction in predictions:
				for i in range(len(prediction)):
					ultimate[i][np.argmax(prediction[i])] += 1
			predicted = []
			for u in ultimate:
				voted = sorted(u, key=u.get, reverse=True)
				predicted.append(voted[0])
			predicted = keras.utils.to_categorical(np.array(predicted), self.n_classes)
			return predicted
		else:
			predicted = np.argmax(np.sum(np.array(predictions),axis=0),axis=1)
			predicted = keras.utils.to_categorical(np.array(predicted), self.n_classes)
                        return predicted

def main(argv=None):

	if FLAGS.dataset not in ['cifar10', 'mnist', 'svhn']:
		print "Invalid dataset specified. Exiting"
		exit()

	bag = Bagging(10, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	#Training mode
	if FLAGS.mode in ['train', 'finetune']:
		# Load model
	        model = load_model(FLAGS.seed_model)
		model.optimizer.lr.assign(FLAGS.learning_rate)

		# Initialize data object
	        dataObject = data_load.get_appropriate_data(FLAGS.dataset)(np.load(FLAGS.data_x), np.load(FLAGS.data_y))

		# Black-box network
	        (blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()

		# Train data
		bag.train(blackbox_Xtrain, blackbox_Ytrain, dataObject, model)

		# Compute bag-level test accuracy
		predicted = np.argmax(bag.predict(FLAGS.model_dir, X_test, FLAGS.predict_mode),1)
		true = np.argmax(Y_test,1)
		acc = (100*(predicted==true).sum()) / float(len(Y_test))
		print "Bag level test accuracy", acc

		#Save model
		model.save(FLAGS.seed_model)

	#Testing mode
	elif FLAGS.mode == 'test':
		X = np.load(FLAGS.data_x)
		Y = np.load(FLAGS.data_y)
		predicted = np.argmax(bag.predict(FLAGS.model_dir, X, FLAGS.predict_mode),1)
		Y = np.argmax(Y, 1)
		acc = (100*(predicted==Y).sum()) / float(len(Y))
		print "Misclassification accuracy",(100-acc)
	else:
		print "Invalid option"


if __name__ == '__main__':
	app.run()
