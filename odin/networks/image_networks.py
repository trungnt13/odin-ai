from functools import partial
from numbers import Number

import numpy as np
import tensorflow as tf
from odin.backend.keras_helpers import layer2text
from odin.networks.base_networks import (NetworkConfig, SequentialNetwork)
from odin.networks.skip_connection import SkipConnection
from tensorflow.python import keras

__all__ = [
    'create_mnist_autoencoder',
    'create_image_autoencoder',
    'ImageNet',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _nparams(distribution, distribution_kw):
  from odin.bay.distribution_alias import parse_distribution
  distribution, _ = parse_distribution(distribution)
  return int(
      tf.reduce_prod(distribution.params_size(1, **distribution_kw)).numpy())


_CONV = partial(keras.layers.Conv2D, padding="SAME")
_DECONV = partial(keras.layers.Conv2DTranspose, padding="SAME")
_DENSE = partial(keras.layers.Dense, use_bias=True)


class Center0Image(keras.layers.Layer):
  r"""Normalize the image pixel from [0, 1] to [-1, 1]"""

  def call(self, inputs, **kwargs):
    return 2. * inputs - 1.


# ===========================================================================
# Basic Network
# ===========================================================================
def create_mnist_autoencoder(latent_size=10,
                             base_depth=32,
                             n_channels=1,
                             activation='relu',
                             center0=True,
                             distribution='bernoulli',
                             distribution_kw=dict()):
  r""" Specialized autoencoder configuration for Binarized MNIST """
  n_params = _nparams(distribution, distribution_kw)
  image_shape = (28, 28, n_channels)
  conv = partial(keras.layers.Conv2D, padding="SAME", activation=activation)
  deconv = partial(keras.layers.Conv2DTranspose,
                   padding="SAME",
                   activation=activation)
  start = [keras.layers.InputLayer(input_shape=image_shape)]
  if center0:
    start.append(Center0Image())

  encoder_net = keras.Sequential(
      start + [
          conv(base_depth, 5, 1),
          conv(base_depth, 5, 2),
          conv(2 * base_depth, 5, 1),
          conv(2 * base_depth, 5, 2),
          conv(4 * latent_size, 7, padding="VALID"),
          keras.layers.Flatten(),
          keras.layers.Dense(2 * latent_size, activation=None),
      ],
      name="Encoder",
  )
  # Collapse the sample and batch dimension and convert to rank-4 tensor for
  # use with a convolutional decoder network.
  decoder_net = keras.Sequential(
      [
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
          keras.layers.Flatten(),
      ],
      name="Decoder",
  )
  return encoder_net, decoder_net


def create_image_autoencoder(image_shape=(64, 64, 1),
                             projection_dim=256,
                             activation='relu',
                             center0=True,
                             distribution='bernoulli',
                             distribution_kw=dict(),
                             skip_connect=False,
                             convolution=True,
                             input_shape=None):
  r""" Initialized the Convolutional encoder and decoder often used in
  Disentangled VAE literatures.

  By default, the image_shape and channels are configurated for binarized MNIST

  Arguments:
    image_shape : tuple of Integer. The shape of input and output image
    input_shape : tuple of Integer (optional). The `input_shape` to the encoder
      is different from the `image_shape` (in case of conditional VAE).
  """
  kw = dict(locals())
  encoder = ImageNet(**kw, decoding=False)
  decoder = ImageNet(**kw, decoding=True)
  return encoder, decoder


# ===========================================================================
# Decoder
# ===========================================================================
class ImageNet(keras.Model):
  r"""
  Arguments:
    image_shape : tuple of Integer. The shape of input and output image
    input_shape : tuple of Integer (optional). The `input_shape` to the encoder
      is different from the `image_shape` (in case of conditional VAE).

  Reference:
    Dieng, A.B., Kim, Y., Rush, A.M., Blei, D.M., 2018. "Avoiding Latent
      Variable Collapse With Generative Skip Models".
      arXiv:1807.04863 [cs, stat].
  """

  def __init__(self,
               image_shape=(28, 28, 1),
               projection_dim=256,
               activation='relu',
               center0=True,
               distribution='bernoulli',
               distribution_kw=dict(),
               skip_connect=False,
               convolution=True,
               decoding=False,
               input_shape=None,
               name=None):
    if name is None:
      name = "Decoder" if decoding else "Encoder"
    super().__init__(name=name)
    if isinstance(image_shape, Number):
      image_shape = (image_shape,)
    ## check multi-inputs
    self.image_shape = [image_shape]
    # input_shape to the encoder is the same as output_shape in the decoder
    if input_shape is None:
      input_shape = image_shape
    # others
    self.skip_connect = bool(skip_connect)
    self.convolution = bool(convolution)
    self.is_mnist = False
    self.pool_size = []
    self.decoding = decoding
    ## prepare layers
    layers = []
    if center0 and not decoding:
      layers.append(Center0Image())
    n_params = _nparams(distribution, distribution_kw)
    ## Dense
    if not convolution:
      if decoding:
        layers += [_DENSE(1000, activation=activation) for i in range(5)] + \
          [_DENSE(int(np.prod(image_shape) * n_params), activation='linear'),
           keras.layers.Reshape(image_shape)]
      else:
        layers += [keras.layers.Flatten()] + \
          [_DENSE(1000, activation=activation) for i in range(5)] + \
          [keras.layers.Dense(projection_dim, use_bias=True, activation='linear')]
    ## MNIST
    elif image_shape[:2] == (28, 28):
      base_depth = 32
      self.is_mnist = True
      if decoding:
        layers = [
            _DENSE(projection_dim, activation=activation),
            keras.layers.Lambda(
                lambda codes: tf.reshape(codes, (-1, 1, 1, projection_dim))),
            _DECONV(2 * base_depth, 7, padding="VALID", activation=activation),
            _DECONV(2 * base_depth, 5, activation=activation),
            _DECONV(2 * base_depth, 5, 2, activation=activation),
            _DECONV(base_depth, 5, activation=activation),
            _DECONV(base_depth, 5, 2, activation=activation),
            _DECONV(base_depth, 5, activation=activation),
            _CONV(image_shape[-1] * n_params, 5, activation='linear'),
            keras.layers.Flatten(),
        ]
      else:
        layers += [
            _CONV(base_depth, 5, 1, activation=activation),
            _CONV(base_depth, 5, 2, activation=activation),
            _CONV(2 * base_depth, 5, 1, activation=activation),
            _CONV(2 * base_depth, 5, 2, activation=activation),
            _CONV(4 * base_depth, 7, padding="VALID", activation=activation),
            keras.layers.Flatten(),
            keras.layers.Dense(projection_dim,
                               use_bias=True,
                               activation='linear')
        ]
        self.pool_size = [1, 2, 2, 4, 28]
    ## Other, must be power of 2
    else:
      assert all(int(np.log2(s)) == np.log2(s) for s in image_shape[:2]), \
        "Image sizes must be power of 2"
      if decoding:
        size = image_shape[1] // 16
        encoder_shape = (size, size, 64)
        layers = [
            _DENSE(projection_dim, activation=activation),
            _DENSE(int(np.prod(encoder_shape)), activation=activation),
            keras.layers.Reshape(encoder_shape),
            _DECONV(64, 4, 2, activation=activation),
            _DECONV(32, 4, 2, activation=activation),
            _DECONV(32, 4, 2, activation=activation),
            _DECONV(image_shape[-1] * n_params, 4, 2, activation='linear'),
            keras.layers.Flatten(),
        ]
      else:
        layers += [
            _CONV(32, 4, 2, activation=activation),
            _CONV(32, 4, 2, activation=activation),
            _CONV(64, 4, 2, activation=activation),
            _CONV(64, 4, 2, activation=activation),
            keras.layers.Flatten(),
            keras.layers.Dense(projection_dim,
                               use_bias=True,
                               activation='linear')
        ]
        self.pool_size = [2, 4, 8, 16]
    ## save the layers
    self._layers = layers
    ## build the network
    if not decoding:
      x = keras.layers.Input(shape=input_shape)
      self(x)

  def __repr__(self):
    return layer2text(self)

  def __str__(self):
    return layer2text(self)

  def call(self, inputs, training=None, mask=None):
    first_inputs = inputs
    if not self.convolution:  # dense
      if not self.decoding:
        first_inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
    else:  # convolution
      if self.decoding:
        first_inputs = tf.expand_dims(inputs, axis=-2)
        first_inputs = tf.expand_dims(first_inputs, axis=-2)
    # iterate each layer
    pool_idx = 0
    for layer_idx, layer in enumerate(self._layers):
      outputs = layer(inputs, training=training)
      inputs = outputs
      ### skip connection
      if (self.skip_connect and
          isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D))):
        ## convolution, downsampling
        if self.convolution and isinstance(layer, keras.layers.Conv2D):
          if self.decoding:  # decoder
            if layer_idx == len(self._layers) - 2:  # skip last layer
              continue
            h, w = inputs.shape[-3:-1]
            batch_shape = (1,) * (len(inputs.shape) - 3)
            inputs = tf.concat(
                [inputs, tf.tile(first_inputs, batch_shape + (h, w, 1))],
                axis=-1,
            )
          else:  # encoder
            p = self.pool_size[pool_idx]
            pool_idx += 1
            inputs = tf.concat(
                [
                    inputs,
                    tf.nn.avg_pool2d(first_inputs, (p, p), (p, p), "SAME")
                ],
                axis=-1,
            )
        ## dense layers
        elif not self.convolution:
          if self.decoding and layer_idx == len(self._layers) - 2:
            continue
          inputs = tf.concat([inputs, first_inputs], axis=-1)
    return outputs
