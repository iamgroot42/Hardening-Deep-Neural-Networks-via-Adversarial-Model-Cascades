import numpy as np

import keras
from keras.models import load_model

from keras.objectives import categorical_crossentropy
from tensorflow.python.platform import app
from keras.utils import np_utils

import utils_cifar, utils_mnist, utils_svhn
import helpers
import os

import tensorflow as tf
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs')
flags.DEFINE_float('sample_ratio', 0.75, 'Percentage of sample to be taken per model for training')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('mode', 'finetune', '(test,finetune)')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_string('model_dir', './', 'path to output directory of models')
flags.DEFINE_string('seed_model', ' ', 'path to seed model')
flags.DEFINE_string('data_x', './', 'path to numpy file of data for prediction')
flags.DEFINE_string('data_y', './', 'path to numpy file of labels for prediction')
flags.DEFINE_float('learning_rate', 0.0001 ,'Learning rate for classifier')
flags.DEFINE_string('predict_mode', 'voting', 'Method for prediction while testing (voting/weighted)')


class Bagging:
	def __init__(self, n_classes, sample_ratio, batch_size, nb_epochs):
		self.n_classes = n_classes
		self.sample_ratio = sample_ratio
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs

	def train(self, X, Y, model):
		subset = np.random.choice(len(Y), int(len(Y) * self.sample_ratio))
		x_sub = X[subset]
		y_sub = Y[subset]
		if FLAGS.dataset == 'cifar100':
			datagen = utils_cifar.augmented_data(x_sub)
		elif FLAGS.dataset == 'svhn':
			datagen = utils_svhn.augmented_data(x_sub)
		X_tr, y_tr, X_val, y_val = helpers.validation_split(x_sub, y_sub, 0.2)
		if FLAGS.dataset != 'mnist':
			model.fit_generator(datagen.flow(X_tr, y_tr,
			  				batch_size=self.batch_size),
							steps_per_epoch=X_tr.shape[0] // self.batch_size,
							epochs=self.nb_epochs,
							validation_data=(X_val, y_val))
		else:
			model.fit(X_tr, y_tr, batch_size = self.batch_size, epochs=self.nb_epochs, validation_data=(X_val, y_val))
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
	bag = None
	n_classes = 10
	tf.set_random_seed(1234)

	if FLAGS.dataset == 'cifar100':
		bag = Bagging(100, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)
		n_classes = 100
	elif FLAGS.dataset == 'mnist':
		bag = Bagging(10, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)
	elif FLAGS.dataset == 'svhn':
		bag = Bagging(10, FLAGS.sample_ratio, FLAGS.batch_size, FLAGS.nb_epochs)
	else:
		print "Invalid dataset specified. Exiting"
		exit()

	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	#Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	#Training mode
	if FLAGS.mode in ['train', 'finetune']:
		# Load model
	        model = load_model(FLAGS.seed_model)

		# Make only dense layers trainable for finetuning
	        for layer in model.layers:
        	        if "dense" not in layer.name:
                	        layer.trainable=False

		# Load data
		if FLAGS.dataset == 'cifar100':
			X, Y, _, _ = utils_cifar.data_cifar()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, 100)
		elif FLAGS.dataset == 'mnist':
			X, Y, _, _ = utils_mnist.data_mnist()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 5000, 10)
		else:
			X, Y, _, _ = utils_svhn.data_svhn()
			X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 4000, 10)
		X_train_p = np.concatenate((X_train_p, np.load(FLAGS.data_x)))
		Y_train_p = np.concatenate((Y_train_p, np.load(FLAGS.data_y)))

		# Train data
		bag.train(X_train_p, Y_train_p, model)

		# Print validation accuracy
		X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
		predicted = np.argmax(bag.predict(FLAGS.model_dir, X_val, FLAGS.predict_mode),1)
		true = np.argmax(y_val,1)
		acc = (100*(predicted==true).sum()) / float(len(y_val))
		print "Whole bag's validation accuracy", acc

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
