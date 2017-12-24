import common
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras
import tensorflow as tf

import data_load

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_string('model_path', 'PM', 'Path where model is stored')
flags.DEFINE_string('dataset', 'cifar10', '(cifar10,svhn,mnist)')

def main(argv=None):
	# Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'th':
                keras.backend.set_image_dim_ordering('th')

	# Initialize data object
	keras.layers.core.K.set_learning_phase(0)
	dataObject = data_load.get_appropriate_data(FLAGS.dataset)()

	if dataObject is None:
		print "Invalid dataset; exiting"
		exit()

	(blackbox_Xtrain, blackbox_Ytrain), (X_test, Y_test) = dataObject.get_blackbox_data()

	model = keras.models.load_model(FLAGS.model_path)

	# Evaluate the accuracy of the blackbox model on adversarial examples
	accuracy = model.evaluate(X_test, Y_test, batch_size=FLAGS.batch_size)
	print('\nTest accuracy: ' + str(accuracy[1]))

	y_model = np.argmax(model.predict(X_test), axis=1)
	y_true = np.argmax(Y_test, axis=1)
	acc = (y_model==y_true).sum()
	acc /= float(len(y_true))
	print('Calculated other way (should be exactly same: ' + str(acc))
	print('Length:  ' + str(len(y_true)))
	print model.predict(X_test)

if __name__ == '__main__':
	app.run()
