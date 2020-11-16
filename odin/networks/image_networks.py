from functools import partial
from numbers import Number
from typing import Callable, Dict, List, Union

import numpy as np
import tensorflow as tf
from odin.backend.keras_helpers import layer2text
from odin.networks.base_networks import NetworkConfig, SequentialNetwork
from odin.networks.skip_connection import SkipConnection
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer

__all__ = [
    'mnist_networks', 'dsprites_networks', 'shapes3d_networks',
    'celeba_networks'
]


# ===========================================================================
# Helpers
# ===========================================================================
class CenterAt0(keras.layers.Layer):
  """Normalize the image pixel from [0, 1] to be centerized
  at 0 given range [-1, 1]
  """

  def __init__(self, enable: bool = True, name: str = 'CenterAt0'):
    super().__init__(name=name)
    self.enable = bool(enable)

  def call(self, inputs, **kwargs):
    if self.enable:
      return 2. * inputs - 1.
    return inputs


def _prepare_cnn(activation=tf.nn.leaky_relu):
  # he_uniform is better for leaky_relu
  if activation is tf.nn.leaky_relu:
    init = 'he_uniform'
  else:
    init = 'glorot_uniform'
  conv = partial(keras.layers.Conv2D,
                 padding='same',
                 kernel_initializer=init,
                 activation=activation)
  deconv = partial(keras.layers.Conv2DTranspose,
                   padding='same',
                   kernel_initializer=init,
                   activation=activation)
  return conv, deconv


class SkipSequential(keras.Model):

  def __init__(self, layers=[], name='SkipGenerator'):
    super().__init__(name=name)
    self.all_layers = list(layers)
    self.proj_layers = list()
    self.activation = list()
    linear = keras.activations.get('linear')
    for l in layers:
      if isinstance(l, keras.layers.Conv2DTranspose):
        self.activation.append(l.activation)
        self.proj_layers.append(
            keras.layers.Conv2D(l.filters, (1, 1),
                                padding='same',
                                activation='linear',
                                name=f'{l.name}_proj'))
        l.activation = linear
      else:
        self.proj_layers.append(None)
        self.activation.append(None)

  def build(self, input_shape):
    # this is a simple logic why keras don't do this by default!
    self._build_input_shape = input_shape
    return super().build(input_shape)

  @property
  def input_shape(self):
    return self._build_input_shape

  def call(self, x, **kwargs):
    z = tf.reshape(x, (-1, 1, 1, x.shape[-1]))
    for fn, proj, activation in zip(self.all_layers, self.proj_layers,
                                    self.activation):
      x = fn(x, **kwargs)
      # shortcut connection
      if proj is not None:
        z_proj = proj(z, **kwargs)
        x = activation(x + z_proj)
    return x


# ===========================================================================
# Basic Network
# ===========================================================================
def mnist_networks(qz: str = 'mvndiag',
                   zdim: int = 32,
                   activation: Union[Callable, str] = tf.nn.leaky_relu,
                   is_semi_supervised: bool = False,
                   centerize_image: bool = True,
                   skip_generator: bool = False,
                   n_channels: int = 1) -> Dict[str, Layer]:
  from odin.bay.random_variable import RVmeta
  input_shape = (28, 28, n_channels)
  conv, deconv = _prepare_cnn(activation=activation)
  encoder = keras.Sequential(
      [
          CenterAt0(enable=centerize_image),
          conv(32, 5, strides=1, name='encoder0'),
          conv(32, 5, strides=2, name='encoder1'),
          conv(64, 5, strides=1, name='encoder2'),
          conv(64, 5, strides=2, name='encoder3'),
          conv(4 * zdim, 7, strides=1, padding='valid', name='encoder4'),
          keras.layers.Flatten()
      ],
      name='encoder',
  )
  layers = [
      keras.layers.Lambda(  # assume that n_mcmc_sample=()
          lambda x: tf.reshape(x, [-1, 1, 1, x.shape[-1]])),
      deconv(64, 7, strides=1, padding='valid', name='decoder0'),
      deconv(64, 5, strides=1, name='decoder1'),
      deconv(64, 5, strides=2, name='decoder2'),
      deconv(32, 5, strides=1, name='decoder3'),
      deconv(32, 5, strides=2, name='decoder4'),
      deconv(32, 5, strides=1, name='decoder5'),
      conv(1, 5, strides=1, activation=None, name='decoder6'),
      keras.layers.Flatten()
  ]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='skipdecoder')
  else:
    decoder = keras.Sequential(layers=layers, name='decoder')
  latents = RVmeta((zdim,), qz, projection=True, name="latents")
  observation = RVmeta(input_shape, "bernoulli", projection=False, name="image")
  networks = dict(encoder=encoder,
                  decoder=decoder,
                  observation=observation,
                  latents=latents)
  if is_semi_supervised:
    networks['labels'] = RVmeta(10, 'onehot', projection=True, name='digits')
  return networks


def dsprites_networks(qz: str = 'mvndiag',
                      zdim: int = 32,
                      activation: Union[Callable, str] = tf.nn.leaky_relu,
                      is_semi_supervised: bool = False,
                      centerize_image: bool = True,
                      skip_generator: bool = False,
                      n_channels: int = 1):
  from odin.bay.random_variable import RVmeta
  input_shape = (64, 64, n_channels)
  conv, deconv = _prepare_cnn(activation=activation)
  proj_dim = 128 if n_channels == 1 else 256
  encoder = keras.Sequential(
      [
          CenterAt0(enable=centerize_image),
          conv(32, 4, strides=2, name='encoder0'),
          conv(32, 4, strides=2, name='encoder1'),
          conv(64, 4, strides=2, name='encoder2'),
          conv(64, 4, strides=2, name='encoder3'),
          keras.layers.Flatten(),
          keras.layers.Dense(proj_dim, activation='linear', name='encoder4')
      ],
      name='encoder',
  )
  layers = [
      keras.layers.Dense(proj_dim, activation=activation, name='decoder0'),
      keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 1, 1, proj_dim))),
      keras.layers.Dense(4 * 4 * 64, activation=activation, name='decoder1'),
      keras.layers.Reshape((4, 4, 64)),
      deconv(64, 4, strides=2, name='decoder2'),
      deconv(32, 4, strides=2, name='decoder3'),
      deconv(32, 4, strides=2, name='decoder4'),
      deconv(n_channels, 4, strides=2, name='decoder5'),
      keras.layers.Flatten()
  ]
  if skip_generator:
    decoder = SkipSequential(layers=layers, name='skipdecoder')
  else:
    decoder = keras.Sequential(layers=layers, name='decoder')
  latents = RVmeta((zdim,), qz, projection=True, name="latents")
  observation = RVmeta(input_shape, "bernoulli", projection=False, name="image")
  return dict(encoder=encoder,
              decoder=decoder,
              observation=observation,
              latents=latents)


def shapes3d_networks(qz: str = 'mvndiag',
                      zdim: int = 32,
                      activation: Union[Callable, str] = tf.nn.leaky_relu,
                      is_semi_supervised: bool = False,
                      centerize_image: bool = True,
                      skip_generator: bool = False,
                      n_channels: int = 3):
  return dsprites_networks(qz=qz,
                           zdim=zdim,
                           activation=activation,
                           is_semi_supervised=is_semi_supervised,
                           centerize_image=centerize_image,
                           skip_generator=skip_generator,
                           n_channels=n_channels)


def celeba_networks(qz: str = 'mvndiag',
                    zdim: int = 32,
                    activation: Union[Callable, str] = tf.nn.leaky_relu,
                    is_semi_supervised: bool = False,
                    centerize_image: bool = True,
                    skip_generator: bool = False,
                    n_channels: int = 3):
  raise NotImplementedError
