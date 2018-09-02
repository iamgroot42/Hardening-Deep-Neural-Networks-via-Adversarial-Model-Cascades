import common
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from art.defences.feature_squeezing import FeatureSqueezing

FLAGS = flags.FLAGS

flags.DEFINE_string('dump_data', "", 'Dat ato dump')
flags.DEFINE_string('load_data', "", 'Data to load')


def main(argv=None):
	# Run FS
	X = np.load(FLAGS.load_data)
	fs = FeatureSqueezing(bit_depth=8)
	squeezed_values = fs(X) 

	# Dump data
	np.save(FLAGS.dump_data, squeezed_values)

if __name__ == "__main__":
	app.run()
