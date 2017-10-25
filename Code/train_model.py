import common

import tensorflow as tf
import numpy as np
import keras

from tensorflow.python.platform import app
from keras.models import load_model
from tensorflow.python.platform import flags
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import utils_mnist, utils_cifar, utils_svhn
from Models import cnn, sota
import helpers


FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', '(train,finetune)')
flags.DEFINE_string('dataset', 'cifar100', '(cifar100,svhn,mnist)')
flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_string('save_here', 'saved_model', 'Path where model is to be saved')
flags.DEFINE_string('level', 'proxy' , '(blackbox,proxy)')
flags.DEFINE_string('proxy_x', 'PX.npy', 'Path where proxy training data is to be saved')
flags.DEFINE_string('proxy_y', 'PY.npy', 'Path where proxy training data labels are to be saved')
flags.DEFINE_integer('proxy_level', 1, 'Model with scale (1,2,4)')
flags.DEFINE_float('label_smooth', 0, 'Amount of label smoothening to be applied')


def main(argv=None):
        n_classes=10
        data_shape = (3,32,32)
        tf.set_random_seed(1234)

        # Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'th':
                keras.backend.set_image_dim_ordering('th')

        # Load dataset
        if FLAGS.level == 'blackbox':
                if FLAGS.dataset == 'cifar100':
                        n_classes = 100
                        X, Y, _, _ = utils_cifar.data_cifar()
                        X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 500, n_classes)
                elif FLAGS.dataset == 'mnist':
                        data_shape = (1,28,28)
                        X, Y, _, _ = utils_mnist.data_mnist()
                        X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 5000, n_classes)
                elif FLAGS.dataset == 'svhn':
                        X, Y, _, _ = utils_svhn.data_svhn()
                        X_train_p, Y_train_p, _,  _ = helpers.jbda(X, Y, "train", 4000, n_classes)
                else:
                        print "Invalid dataset; exiting"
                        exit()
                if FLAGS.mode == 'finetune':
                        adv_x = np.load(FLAGS.proxy_x)
                        adv_y = np.load(FLAGS.proxy_y)
                        X_train_p = np.concatenate((X_train_p, adv_x))
                        Y_train_p = np.concatenate((Y_train_p, adv_y))
        elif FLAGS.level == 'proxy':
                X_train_p = np.load(FLAGS.proxy_x)
                Y_train_p = np.load(FLAGS.proxy_y)
                if FLAGS.dataset == 'mnist':
                        data_shape = (1,28,28)

        # Don't hog GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

        # Black-box network
        if FLAGS.level == 'blackbox':
                X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
		# Early stopping and dynamic lr
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
                early_stop = EarlyStopping(monitor='val_loss', min_delta=0.03, patience=5)
                if FLAGS.label_smooth > 0:
                        y_tr = y_tr.clip(FLAGS.label_smooth / 9., 1. - FLAGS.label_smooth)
                if FLAGS.dataset == 'cifar100':
                        model = sota.cifar_svhn(FLAGS.learning_rate, n_classes)
                        if FLAGS.mode == 'finetune':
                                model = load_model(FLAGS.save_here)
                                model.optimizer.lr.assign(FLAGS.learning_rate)
                        datagen = utils_cifar.augmented_data(X_train_p)
                        model.fit_generator(datagen.flow(X_tr, y_tr,
                                batch_size=FLAGS.batch_size),
                                steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
                                epochs=FLAGS.nb_epochs,
				callbacks=[reduce_lr, early_stop],
                                validation_data=(X_val, y_val))
                elif FLAGS.dataset == 'svhn':
                        model = sota.cifar_svhn(FLAGS.learning_rate, n_classes)
                        if FLAGS.mode == 'finetune':
                                model = load_model(FLAGS.save_here)
                                model.optimizer.lr.assign(FLAGS.learning_rate)
                        datagen = utils_svhn.augmented_data(X_train_p)
                        model.fit_generator(datagen.flow(X_tr, y_tr,
                                batch_size=FLAGS.batch_size),
                                steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
                                epochs=FLAGS.nb_epochs,
				callbacks=[reduce_lr, early_stop],
                                validation_data=(X_val, y_val))
                        accuracy = model.evaluate(X_val, y_val, batch_size=FLAGS.batch_size)
                else:
                        model = sota.mnist(FLAGS.learning_rate, n_classes)
                        if FLAGS.mode == 'finetune':
                                model = load_model(FLAGS.save_here)
                                model.optimizer.lr.assign(FLAGS.learning_rate)
                        model.fit(X_tr, y_tr,
                                batch_size=FLAGS.batch_size,
                                epochs=FLAGS.nb_epochs,
				callbacks=[reduce_lr, early_stop],
                                validation_data=(X_val, y_val))

                accuracy = model.evaluate(X_val, y_val, batch_size=FLAGS.batch_size)
                print('\nTest accuracy for black-box model: ' + str(accuracy[1]*100))
                model.save(FLAGS.save_here)

        # Proxy network
        else:
                model = cnn.proxy(nb_classes=n_classes, learning_rate=FLAGS.learning_rate, shape=data_shape, scale=FLAGS.proxy_level)
                X_tr, y_tr, X_val, y_val = helpers.validation_split(X_train_p, Y_train_p, 0.2)
                if FLAGS.dataset == 'cifar100':
                        datagen = utils_cifar.augmented_data(X_train_p)
                        model.fit_generator(datagen.flow(X_tr, y_tr,
                                batch_size=FLAGS.batch_size),
                                steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
                                epochs=FLAGS.nb_epochs,
                                validation_data=(X_val, y_val))
                elif FLAGS.dataset == 'svhn':
                        datagen = utils_svhn.augmented_data(X_train_p)
                        model.fit_generator(datagen.flow(X_tr, y_tr,
                                batch_size=FLAGS.batch_size),
                                steps_per_epoch=X_tr.shape[0] // FLAGS.batch_size,
                                epochs=FLAGS.nb_epochs,
                                validation_data=(X_val, y_val))
                else:
                        model.fit(X_tr, y_tr, batch_size=FLAGS.batch_size, epochs=FLAGS.nb_epochs, validation_data=(X_val, y_val))
                accuracy = model.evaluate(X_val, y_val, batch_size=FLAGS.batch_size)
                print('\nTest accuracy for proxy model: ' + str(accuracy[1]*100))
                model.save(FLAGS.save_here)


if __name__ == '__main__':
        app.run()
