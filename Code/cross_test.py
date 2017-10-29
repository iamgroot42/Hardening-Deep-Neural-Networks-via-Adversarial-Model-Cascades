import common
import keras

from keras.models import  load_model
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_string('model_path', 'BM', 'Path where model is stored')
flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are saved')


def main(argv=None):
	# Image dimensions ordering should follow the Theano convention
	if keras.backend.image_dim_ordering() != 'th':
		keras.backend.set_image_dim_ordering('th')

	# Load data for testing
	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	model = load_model(FLAGS.model_path)
	accuracy = model.evaluate(X_test_adv, Y_test, batch_size=FLAGS.batch_size)
	print('\nMisclassification accuracy on adversarial examples: ' + str(1.0 - accuracy[1]))


if __name__ == '__main__':
	app.run()
