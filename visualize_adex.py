from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.python.platform import app
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are saved')
flags.DEFINE_string('adversary_path_xo', 'ADXO.npy', 'Path where original examples are saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are saved')
flags.DEFINE_integer('example_index', 0, 'Which index do you want to visualize?')
flags.DEFINE_integer('dataset', 0 , 'MNIST(0), CIFAR10(1)')

def main(argv=None):

	X_test = np.load(FLAGS.adversary_path_xo)
	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	X_test_adv = X_test_adv[FLAGS.example_index]
	X_test = X_test[FLAGS.example_index]
	Y_test = Y_test[FLAGS.example_index]

	if FLAGS.dataset == 0:
		X_test_adv = X_test_adv.reshape((28, 28))
		X_test = X_test.reshape((28, 28))
		plt.matshow(X_test_adv,  cmap='gray')
		plt.savefig('adv_example.png')
		plt.matshow(X_test,  cmap='gray')
		plt.savefig('example.png')
	else:
		X_test_adv = np.reshape(X_test_adv, (32,32,3))
		X_test = np.reshape(X_test, (32,32,3))
		plt.imshow(X_test_adv)
		plt.savefig('adv_example.png')
		plt.imshow(X_test)
		plt.savefig('example.png')


if __name__ == '__main__':
	app.run()
