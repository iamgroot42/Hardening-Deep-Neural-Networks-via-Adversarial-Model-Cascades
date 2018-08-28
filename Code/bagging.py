import common

import numpy as np

import keras
from keras.models import load_model

from keras.objectives import categorical_crossentropy
from tensorflow.python.platform import app
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from cleverhans.utils_keras import KerasModelWrapper

import os

import data_load, helpers
from keras import backend as K

import tensorflow as tf
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('mode', 'finetune', '(test,finetune)')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')
flags.DEFINE_string('model_dir', './', 'path to output directory of models')
flags.DEFINE_string('seed_model', ' ', 'path to seed model')
flags.DEFINE_string('data_x', './', 'path to numpy file of data for prediction')
flags.DEFINE_string('data_y', './', 'path to numpy file of labels for prediction')
flags.DEFINE_float('learning_rate', 0.01 ,'Learning rate for classifier')
flags.DEFINE_string('predict_mode', 'weighted', 'Method for prediction while testing (voting/weighted)')
flags.DEFINE_string('attack', "" , "Attack against which adversarial training is to be done")


class Bagging:
	def __init__(self, n_classes, batch_size, nb_epochs):
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs

	def train(self, X_tr, y_tr, X_val, y_val, dataObject, model):
		# Get and fit generator
		datagen = dataObject.data_generator()
		datagen.fit(X_tr)
		attacks = FLAGS.attack.split(',')
		if len(attacks) > 1:
			attacks = attacks[1:]
			attack_params = []
			clever_wrapper = KerasModelWrapper(model)
			for attack in attacks:
				attack_params.append(helpers.get_appropriate_attack(FLAGS.dataset, dataObject.get_range(), attack,
					clever_wrapper, common.sess, harden=True, attack_type="black"))
		else:
			attack_params=None
		# Scheduler, assuming RESNET
		def scheduler(epoch):
			if epoch < 31:
				return 0.1
			return 0.01
		helpers.customTrainModel(model, X_tr, y_tr, X_val, y_val, datagen, self.nb_epochs, scheduler, self.batch_size, attacks=attack_params)

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

	bag = Bagging(10, FLAGS.batch_size, FLAGS.nb_epochs)

	#Training mode
	if FLAGS.mode in ['finetune']:
		# Load model
	        model = load_model(FLAGS.seed_model)
		K.set_value(model.optimizer.lr, FLAGS.learning_rate)

		# Initialize data object
	        dataObject = data_load.get_appropriate_data(FLAGS.dataset)(None, None)

		# Black-box network
	        (blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()
		(X_val, Y_val) = dataObject.get_validation_data()

		# Train data
		bag.train(blackbox_Xtrain, blackbox_Ytrain, X_val, Y_val, dataObject, model)

		# Compute bag-level test accuracy
		predicted = np.argmax(bag.predict(FLAGS.model_dir, X_test, FLAGS.predict_mode),1)
		true = np.argmax(Y_test,1)
		acc = (100*(predicted==true).sum()) / float(len(Y_test))
		print("Bag level test accuracy: %f\n" % acc)

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
