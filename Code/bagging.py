import numpy as np

import keras
from keras.models import load_model

from keras.objectives import categorical_crossentropy
from keras.utils import np_utils

import utils_cifar
import helpers
import os

class Bagging:
	def __init__(self, n_classes, sample_ratio, batch_size, nb_epochs):
		self.n_classes = n_classes
		self.sample_ratio = sample_ratio
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.models = []

	def load_models(self, data_dir):
		for file in os.listdir(data_dir):
			self.models.append(load_model(data_dir + file))

	def train(self, X, Y, data_dir, finetune=False):
		if finetune:
			self.load_models(data_dir)
		subsets = []
		for i in range(len(self.models)):
			subsets.append(np.random.choice(len(Y), int(len(Y) * self.sample_ratio)))
		for i, subset in enumerate(subsets):
			x_sub = X[subset]
			y_sub = Y[subset]
			datagen = utils_cifar.augmented_data(x_sub)
			X_tr, y_tr, X_val, y_val = helpers.validation_split(x_sub, y_sub, 0.2)
			self.models[i].fit_generator(datagen.flow(X_tr, y_tr,
                                batch_size=self.batch_size),
                                steps_per_epoch=X_tr.shape[0] // self.batch_size,
                                epochs=self.nb_epochs,
                                validation_data=(X_val, y_val))
			accuracy = self.models[i].evaluate(X_val, y_val, batch_size=self.batch_size)
			print("\nTest accuracy for bag" + str(i) + " model: " + str(accuracy[1]*100))
			self.models[i].save(data_dir + "bag" + str(i))

	def predict(self, predict_on):
		predictions = []
		for model in self.models:
			predictions.append(model.predict(predict_on))
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


if __name__ == "__main__":
	import sys
	bast = Bagging(100, 0.5, 16, int(sys.argv[2]))
	if int(sys.argv[3]) == 1:
		X, Y, _, _ = utils_cifar.data_cifar()
		X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, 100)
		bast.train(X_train_p, Y_train_p, sys.argv[1], True)
		bast.load_models(sys.argv[1])
		predicted = np.argmax(bast.predict(X_train_p),1)
		true = np.argmax(Y_train_p,1)
		acc = (100*(predicted==true).sum()) / float(len(Y_train_p))
		print "Final training accuracy", acc
	elif int(sys.argv[3]) == 2:
		X = np.load(sys.argv[4])
		Y = np.load(sys.argv[5])
		bast.load_models(sys.argv[1])
		predicted = np.argmax(bast.predict(X),1)
		Y = np.argmax(Y, 1)
		acc = (100*(predicted==Y).sum()) / float(len(Y))
		print "Misclassification accuracy",(100-acc)
	elif int(sys.argv[3]) == 4:
		X, Y, _, _ = utils_cifar.data_cifar()
                X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, 100)
		X_train_p = np.concatenate((X_train_p, np.load(sys.argv[4])))
                Y_train_p = np.concatenate((Y_train_p, np.load(sys.argv[5])))
                bast.train(X_train_p, Y_train_p, sys.argv[1], True)
                bast.load_models(sys.argv[1])
	else:
		print "Invalid option"
