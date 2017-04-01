'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics


def get_model(batch_size=100):
	img_rows, img_cols, img_chns = 32, 32, 3
	latent_dim = 10
	intermediate_dim = 128
	epsilon_std = 1.0
	filters = 64
	num_conv = 3
	if K.image_data_format() == 'channels_first':
	   original_img_size = (img_chns, img_rows, img_cols)
	else:
		original_img_size = (img_rows, img_cols, img_chns)
	x = Input(batch_shape=(batch_size,) + original_img_size)
	conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(x)
	conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv_1)
	conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_2)
	conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_3)
	flat = Flatten()(conv_4)
	hidden = Dense(intermediate_dim, activation='relu')(flat)

	z_mean = Dense(latent_dim)(hidden)
	z_log_var = Dense(latent_dim)(hidden)

	def sampling(args):
			z_mean, z_log_var = args
			epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
			return z_mean + K.exp(z_log_var) * epsilon

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
		
	decoder_hid = Dense(intermediate_dim, activation='relu')
	decoder_upsample = Dense(filters * img_rows/2 * img_cols/2, activation='relu')

	if K.image_data_format() == 'channels_first':
			output_shape = (batch_size, filters, img_rows/2, img_cols/2)
	else:
			output_shape = (batch_size, img_rows/2, img_cols/2, filters)

	decoder_reshape = Reshape(output_shape[1:])
	decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
	decoder_deconv_2 = Conv2DTranspose(filters, num_conv, padding='same', strides=1, activation='relu')

	if K.image_data_format() == 'channels_first':
			output_shape = (batch_size, filters, 33, 33)
	else:
			output_shape = (batch_size, img_rows+1, img_cols+1, filters)

	decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
	decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')

	hid_decoded = decoder_hid(z)
	up_decoded = decoder_upsample(hid_decoded)
	reshape_decoded = decoder_reshape(up_decoded)
	deconv_1_decoded = decoder_deconv_1(reshape_decoded)
	deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
	x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
	x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

	def vae_loss(x, x_decoded_mean):
		x = K.flatten(x)
		x_decoded_mean = K.flatten(x_decoded_mean)
		xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean)
		kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return xent_loss + kl_loss

	vae = Model(x, x_decoded_mean_squash)
	vae.compile(optimizer='rmsprop', loss=vae_loss)
	return vae
