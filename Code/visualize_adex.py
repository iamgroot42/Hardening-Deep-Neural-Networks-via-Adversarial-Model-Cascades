import common

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
flags.DEFINE_integer('example_index', 3, 'Which index do you want to visualize?')
flags.DEFINE_integer('dataset', 0 , 'MNIST(0), CIFAR10/100(1)')

def main(argv=None):

	#X_test = np.load(FLAGS.adversary_path_xo)
	X_test_adv = np.load(FLAGS.adversary_path_x)
	Y_test = np.load(FLAGS.adversary_path_y)

	X_test_adv = X_test_adv[FLAGS.example_index]
	#X_test = X_test[FLAGS.example_index]
	Y_test = Y_test[FLAGS.example_index]

	if FLAGS.dataset == 0:
		plt.matshow(X_test_adv,  cmap='gray')
		plt.savefig('adv_example.png')
		plt.matshow(X_test,  cmap='gray')
		plt.savefig('example.png')
	else:
		print("wut")
		X_test_adv = np.swapaxes(X_test_adv,0,2)
		X_test_adv = np.swapaxes(X_test_adv,0,1)
		#X_test = np.swapaxes(X_test,0,2)
		#X_test = np.swapaxes(X_test,0,1)
		plt.imshow(X_test_adv)
		plt.savefig('adv_example.png')
		print("saved")
		#X_test *= 128
		#X_test += 128
		#X_test /= 255
		#plt.imshow(X_test)
		#plt.savefig('example.png')


if __name__ == '__main__':
	app.run()
