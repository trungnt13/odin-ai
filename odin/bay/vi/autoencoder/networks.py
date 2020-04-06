from functools import partial

import tensorflow as tf
from tensorflow.python import keras

from odin.networks import (ConvNetwork, DeconvNetwork, DenseNetwork,
                           NetworkConfig)

__all__ = ['create_mnist_autoencoder', 'create_image_autoencoder']


def create_mnist_autoencoder(latent_dim=10, base_depth=32, activation='relu'):
  r""" Common autoencoder configuration for Binarized MNIST """
  image_shape = (28, 28, 1)
  conv = partial(keras.layers.Conv2D, padding="SAME", activation=activation)

  encoder_net = keras.Sequential([
      conv(base_depth, 5, 1, input_shape=image_shape),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 1),
      conv(2 * base_depth, 5, 2),
      conv(4 * latent_dim, 7, padding="VALID"),
      keras.layers.Flatten(),
      keras.layers.Dense(2 * latent_dim, activation=None),
  ],
                                 name="EncoderNet")

  deconv = partial(keras.layers.Conv2DTranspose,
                   padding="SAME",
                   activation=activation)
  conv = partial(keras.layers.Conv2D, padding="SAME", activation=activation)
  # Collapse the sample and batch dimension and convert to rank-4 tensor for
  # use with a convolutional decoder network.
  decoder_net = keras.Sequential([
      keras.layers.Lambda(lambda codes: tf.reshape(codes,
                                                   (-1, 1, 1, latent_dim)),
                          batch_input_shape=(None, latent_dim)),
      deconv(2 * base_depth, 7, padding="VALID"),
      deconv(2 * base_depth, 5),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5),
      conv(image_shape[-1], 5, activation=None),
  ],
                                 name="DecoderNet")
  return encoder_net, decoder_net


def create_image_autoencoder(image_shape=(28, 28),
                             channels=1,
                             latent_dim=10,
                             projection_dim=256,
                             activation='relu',
                             batchnorm=False,
                             distribution='diag',
                             distribution_kw=dict(),
                             conv=True,
                             name="Image"):
  r""" Initialized the Convolutional encoder and decoder often used in
  Disentangled VAE literatures.

  By default, the image_shape and channels are configurated for binarized MNIST
  """
  from odin.bay import parse_distribution
  distribution, _ = parse_distribution(distribution)
  n_params = int(
      tf.reduce_prod(distribution.params_size(1, **distribution_kw)).numpy())

  ### Convoltional networks
  if conv:
    encoder = ConvNetwork(filters=[32, 32, 64, 64],
                          kernel_size=[4, 4, 4, 4],
                          strides=[2, 2, 2, 2],
                          batchnorm=bool(batchnorm),
                          activation=activation,
                          end_layers=[
                              keras.layers.Flatten(),
                              keras.layers.Dense(projection_dim,
                                                 activation='linear')
                          ],
                          input_shape=image_shape + (channels,),
                          name="%sEncoder" % name)
    encoder_shape = encoder.layers[-3].output.shape[1:]
    decoder = DeconvNetwork(filters=[64, 32, 32, channels * n_params],
                            kernel_size=[4, 4, 4, 4],
                            strides=[2, 2, 2, 2],
                            activation=[activation] * 3 + ['linear'],
                            batchnorm=bool(batchnorm),
                            start_layers=[
                                keras.layers.Dense(256, activation='relu'),
                                keras.layers.Dense(int(np.prod(encoder_shape)),
                                                   activation='relu'),
                                keras.layers.Reshape(encoder_shape),
                            ],
                            end_layers=[keras.layers.Flatten()],
                            input_shape=(latent_dim,),
                            name="%sDecoder" % name)
  ### Dense networks
  else:
    nlayers = 6
    units = [1000] * nlayers
    encoder = DenseNetwork(units=units,
                           batchnorm=bool(batchnorm),
                           activation=activation,
                           flatten=True,
                           input_shape=image_shape + (channels,),
                           name="%sEncoder" % name)
    decoder = DenseNetwork(
        units=units + [int(np.prod(image_shape) * channels * n_params)],
        batchnorm=bool(batchnorm),
        activation=[activation] * 6 + ['linear'],
        input_shape=(latent_dim,),
        name="%sDecoder" % name,
    )
  return encoder, decoder
