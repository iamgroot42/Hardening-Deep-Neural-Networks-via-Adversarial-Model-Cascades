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

flags.DEFINE_string('adversary_path_x', 'ADX.npy', 'Path where adversarial examples are to be saved')
flags.DEFINE_string('adversary_path_y', 'ADY.npy', 'Path where adversarial labels are to be saved')
flags.DEFINE_integer('example_index', 0, 'Which index do you want to visualize?')


def main(argv=None):

	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	X_test_adv = X_test_adv[FLAGS.example_index]
	Y_test = Y_test[FLAGS.example_index]

	X_test_adv = X_test_adv.reshape((28, 28))
	plt.matshow(X_test_adv,  cmap='gray')
	plt.savefig('adv_example.png')


if __name__ == '__main__':
	app.run()
