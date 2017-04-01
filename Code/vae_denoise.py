import common

import keras
import numpy as np

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_tf import batch_eval
import utils_cifar
import helpers
from models import vaenc

from keras.models import load_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_string('input_data', 'ADX.npy', 'Path where noisy data is saved')
flags.DEFINE_string('denoised_data', 'ADX.npy', 'Path where denoised data is to be saved')
flags.DEFINE_boolean('training_stage', False, 'Whether model is being trained or used for denoising')
flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_string('save_here', 'DENOISE', 'Path where model is to be saved')



def main(argv=None):

	if not FLAGS.training_stage:
		model = load_model(FLAGS.save_here)
		X = np.load(FLAGS.input_data)
		X_ = model.predict(X)
		np.save(FLAGS.denoised_data, X_)
	else:
		model = vaenc.get_model(FLAGS.batch_size)
		X_train, Y_train, X_test, Y_test = utils_cifar.data_cifar_raw()
		X_train_bm, Y_train_bm, X_train_pm, Y_train_pm = helpers.jbda(X_train, Y_train, "train", 500, n_classes)
		X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_bm, Y_train_bm, 0.2)

		vae.fit(X_tr, X_tr,
			epochs=FLAGS.nb_epochs,
			batch_size=FLAGS.batch_size,
			validation_data=(X_val, X_val))
		vae.save(FLAGS.save_here)

if __name__ == '__main__':
	app.run()
