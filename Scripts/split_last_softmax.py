import tensorflow as tf
import sys
import keras

from keras.models import load_model, Model
from keras.layers import Activation

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

original = sys.argv[1]

model = load_model(sys.argv[1])
learning_rate = 1e-4

model.layers[-1].activation = keras.activations.linear
new_out = Activation('softmax')(model.output)

new_model = Model(inputs=model.input, outputs=new_out)
sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

new_model.save(original)
