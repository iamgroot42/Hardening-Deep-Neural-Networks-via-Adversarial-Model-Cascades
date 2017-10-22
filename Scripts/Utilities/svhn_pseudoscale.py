import tensorflow as tf
import sys
import keras

from keras.models import load_model, Model
from keras.layers import Lambda, Input

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

original = sys.argv[1]

model = load_model(sys.argv[1])
learning_rate = 1e-4

input = Input(shape=(3, 32, 32))
lamb = Lambda(lambda x: x * 255)(input)

for layer in model.layers:
	lamb = layer(lamb)

new_model = Model(inputs=input, outputs=lamb)
sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
new_model.save(original)
