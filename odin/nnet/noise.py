from __future__ import division, absolute_import

from odin import backend as K
from odin.nnet.base import NNOp

import tensorflow as tf
from tensorflow.python.ops import init_ops


class Dropout(NNOp):
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.


  Parameters
  ----------
  x: A tensor.
      input tensor (any shape)
  level: float(0.-1.)
      probability dropout values in given tensor
  noise_dims: int or list(int)
      these dimensions will be setted to 1 in noise_shape, and
      used to broadcast the dropout mask.
  noise_type: 'gaussian' (or 'normal'), 'uniform'
      distribution used for generating noise
  rescale: bool
      whether rescale the outputs by dividing the retain probablity

  """

  def __init__(self, level=0.5,
               noise_dims=None, noise_type='uniform',
               rescale=True, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    self.level = level
    self.noise_dims = noise_dims
    self.noise_type = noise_type
    self.rescale = rescale

  def _apply(self, X):
    return K.rand.apply_dropout(X, level=self.level,
                           noise_dims=self.noise_dims,
                           noise_type=self.noise_type,
                           rescale=self.rescale)

  def _transpose(self):
    return self


class Noise(NNOp):
  """
  Parameters
  ----------
  x: A tensor.
      input tensor (any shape)
  level : float or tensor scalar
      Standard deviation of added Gaussian noise, or range of
      uniform noise
  noise_dims: int or list(int)
      these dimensions will be setted to 1 in noise_shape, and
      used to broadcast the dropout mask.
  noise_type: 'gaussian' (or 'normal'), 'uniform'
      distribution used for generating noise

  """

  def __init__(self, level=0.075, noise_dims=None,
               noise_type='gaussian', **kwargs):
    super(Noise, self).__init__(**kwargs)
    self.level = level
    self.noise_dims = noise_dims
    self.noise_type = noise_type

  def _apply(self, X):
    # possible mechanism to hard override the noise-state
    return K.rand.apply_noise(X, level=self.level,
                         noise_dims=self.noise_dims,
                         noise_type=self.noise_type)

  def _transpose(self):
    return self


class GaussianDenoising(NNOp):
  """ Gaussian denoising function proposed in
  "Semi-Supervised Learning with Ladder Networks"
  """

  def __init__(self, activation=tf.nn.sigmoid, **kwargs):
    super(GaussianDenoising, self).__init__(**kwargs)
    self.activation = K.linear if activation is None else activation

  def _initialize(self):
    input_shape = self.input_shape
    shape = input_shape[1:]
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a1', roles=Weight)
    # self.config.create_params(
    #     constant(1.), shape=shape, name='a2', roles=Weight)
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a3', roles=Bias)
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a4', roles=Weight)
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a5', roles=Bias)

    # self.config.create_params(
    #     constant(0.), shape=shape, name='a6', roles=Weight)
    # self.config.create_params(
    #     constant(1.), shape=shape, name='a7', roles=Bias)
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a8', roles=Bias)
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a9', roles=Weight)
    # self.config.create_params(
    #     constant(0.), shape=shape, name='a10', roles=Bias)

  def _apply(self, u, mean, std, z_corr):
    mu = self.a1 * self.activation(self.a2 * u + self.a3) + self.a4 * u + self.a5
    v = self.a6 * self.activation(self.a7 * u + self.a8) + self.a9 * u + self.a10

    z_est = (z_corr - mu) * v + mu
    z_est_bn = (z_est - mean) / K.square(std)
    return z_est_bn

  def _transpose(self):
    raise NotImplementedError
