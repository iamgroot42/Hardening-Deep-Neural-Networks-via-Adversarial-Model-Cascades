import keras
from keras.models import load_model, Model
from keras.layers import Activation

import tensorflow as tf
import sys

train_temp=1
learning_rate = 1e-4

def fn(correct, predicted):
	return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted/train_temp)

base = load_model(sys.argv[1], custom_objects={'fn':fn})

logit = base.output
softmax = Activation('softmax', name='self_added_softmax')(logit)
new_model = Model(base.input, softmax)

sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
new_model.save(sys.argv[2])

