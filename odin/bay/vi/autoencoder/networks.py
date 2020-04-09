from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from odin.networks import (ConvNetwork, DeconvNetwork, DenseNetwork,
                           NetworkConfig)

__all__ = ['create_mnist_autoencoder', 'create_image_autoencoder']


def _nparams(distribution, distribution_kw):
  from odin.bay import parse_distribution
  distribution, _ = parse_distribution(distribution)
  return int(
      tf.reduce_prod(distribution.params_size(1, **distribution_kw)).numpy())


def create_mnist_autoencoder(image_shape=(28, 28, 1),
                             latent_size=10,
                             base_depth=32,
                             activation='relu',
                             center0=True,
                             distribution='bernoulli',
                             distribution_kw=dict()):
  r""" Common autoencoder configuration for Binarized MNIST """
  n_params = _nparams(distribution, distribution_kw)
  conv = partial(keras.layers.Conv2D, padding="SAME", activation=activation)
  deconv = partial(keras.layers.Conv2DTranspose,
                   padding="SAME",
                   activation=activation)
  start = [keras.layers.InputLayer(input_shape=image_shape)]
  if center0:
    start.append(keras.layers.Lambda(lambda inputs: 2 * inputs - 1))

  encoder_net = keras.Sequential(start + [
      conv(base_depth, 5, 1),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 1),
      conv(2 * base_depth, 5, 2),
      conv(4 * latent_size, 7, padding="VALID"),
      keras.layers.Flatten(),
      keras.layers.Dense(2 * latent_size, activation=None),
  ],
                                 name="Encoder")
  # Collapse the sample and batch dimension and convert to rank-4 tensor for
  # use with a convolutional decoder network.
  decoder_net = keras.Sequential([
      keras.layers.Lambda(lambda codes: tf.reshape(codes,
                                                   (-1, 1, 1, latent_size)),
                          batch_input_shape=(None, latent_size)),
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(image_shape[-1] * n_params, 5, activation=None),
  ],
                                 name="Decoder")
  return encoder_net, decoder_net


def create_image_autoencoder(image_shape=(28, 28, 1),
                             latent_size=10,
                             projection_dim=256,
                             activation='relu',
                             center0=True,
                             distribution='bernoulli',
                             distribution_kw=dict(),
                             conv=True):
  r""" Initialized the Convolutional encoder and decoder often used in
  Disentangled VAE literatures.

  By default, the image_shape and channels are configurated for binarized MNIST
  """
  n_params = _nparams(distribution, distribution_kw)
  if len(image_shape) == 2:
    image_shape = list(image_shape) + [1]
  channels = image_shape[-1]

  start = [keras.layers.InputLayer(input_shape=image_shape)]
  if center0:
    start.append(keras.layers.Lambda(lambda inputs: 2 * inputs - 1))
  ### Convoltional networks
  if conv:
    conv = partial(keras.layers.Conv2D,
                   padding="SAME",
                   use_bias=True,
                   activation=activation)
    deconv = partial(keras.layers.Conv2DTranspose,
                     padding="SAME",
                     activation=activation)
    encoder = keras.Sequential(start + [
        conv(32, 4, 2),
        conv(32, 4, 2),
        conv(64, 4, 2),
        conv(64, 4, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(projection_dim, use_bias=True, activation='linear')
    ],
                               name="Encoder")
    encoder_shape = encoder.layers[-3].output.shape[1:]
    decoder = keras.Sequential([
        keras.layers.Dense(
            projection_dim, activation=activation, input_shape=(latent_size,)),
        keras.layers.Dense(int(np.prod(encoder_shape)), activation=activation),
        keras.layers.Reshape(encoder_shape),
        deconv(64, 4, 2),
        deconv(32, 4, 2),
        deconv(32, 4, 2),
        deconv(channels * n_params, 4, 2, activation='linear'),
    ],
                               name="Decoder")
  ### Dense networks
  else:
    nlayers = 6
    units = [1000] * nlayers
    dense = partial(keras.layers.Dense, use_bias=True, activation=activation)
    encoder = keras.Sequential(start + [keras.layers.Flatten()] +
                               [dense(i) for i in units],
                               name="Encoder")
    decoder = keras.Sequential(
        [keras.layers.InputLayer(input_shape=(latent_size,))] +
        [dense(i) for i in units] +
        [dense(int(np.prod(image_shape) * n_params), activation='linear'),
         keras.layers.Reshape(image_shape)],
        name="Decoder")
  return encoder, decoder
