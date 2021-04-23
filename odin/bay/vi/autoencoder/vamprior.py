from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from odin.backend.interpolation import Interpolation, linear
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.utils import as_tuple
from tensorflow import keras
from tensorflow_probability.python.distributions import (NOT_REPARAMETERIZED,
                                                         Distribution, Normal)

__all__ = [
  'Vamprior',
  'VampriorVAE'
]


def hard_probs(x):
  return tf.clip_by_value(x, 0.0, 1.0)


class Vamprior(Distribution):
  """ The distribution represent Variational mixture of prior (Vamprior)
  that formulated from pseudoinputs

  Parameters
  ----------
  input_shape : List[int]
      input shape to the VAE
  fn_encoding : Callable[[tf.Tensor], Distribution]
      the encoding function of VAE, take image as inputs and return the
      posterior latents distribution `q(z|x)`
  n_components : int, optional
      number of mixture components for pseudo-inputs,
      by default 500
  pseudoinputs : Optional[Union[tf.Tensor, np.ndarray]]
      training examples could be pseudoinputs (if given). The shape
      must be `[n_components, np.prod(input_shape)]`, by default None
  pseudoinputs_mean : float, optional
      mean of the pseudo-inputs, by default -0.05
  pseudoinputs_std : float, optional
      standard deviation of the pseudo-inputs, by default 0.01
  dtype : str, optional
      dtype for the distribution, by default 'float32'
  name : str, optional
      name for the distribution, by default 'Vamprior'

  References
  ----------
  Tomczak, J.M., Welling, M., 2018. VAE with a VampPrior. arXiv:1705.07120 [cs, stat].

  """

  def __init__(self,
               input_shape: List[int],
               fn_encoding: Callable[[tf.Tensor], Distribution],
               n_components: int = 500,
               pseudoinputs: Optional[Union[tf.Tensor, np.ndarray]] = None,
               pseudoinputs_mean: float = -0.05,
               pseudoinputs_std: float = 0.01,
               dtype: str = 'float32',
               name: str = 'Vamprior'):
    super().__init__(dtype=dtype,
                     reparameterization_type=NOT_REPARAMETERIZED,
                     validate_args=False,
                     allow_nan_stats=True,
                     parameters={},
                     name=name)
    self.n_components = n_components
    self.means = keras.layers.Dense(
      units=int(np.prod(input_shape)),
      use_bias=False,
      kernel_initializer=tf.initializers.RandomNormal(
        mean=pseudoinputs_mean, stddev=pseudoinputs_std),
      activation=hard_probs)
    self.means.build((None, n_components))
    if pseudoinputs is not None:
      self.means.kernel.assign(pseudoinputs)
    # create an idle input for calling pseudo-inputs
    self.idle_input = tf.eye(n_components, n_components, dtype=dtype)
    self.C = tf.math.log(tf.constant(self.n_components, dtype=dtype))
    self.fn_encoding = fn_encoding
    self.input_shape = as_tuple(input_shape, t=int)

  def _log_prob(self, z):
    # calculate params
    X = self.means(self.idle_input)  # [n_components, input_size]
    X = tf.reshape(X, (-1,) + self.input_shape)
    # calculate params for given data
    qz = self.fn_encoding(X)  # [n_components, zdim]
    z = tf.expand_dims(z, 1)  # [batch_dim, 1, zdim]
    a = qz.log_prob(z) - self.C  # [batch_dim, n_components]
    a = tf.reduce_logsumexp(a, axis=-1)  # [batch_dim]
    return a

  def _sample_n(self, n, seed=None):
    means = self.means(self.idle_input)[:n]  # [n, input_size]
    means = tf.reshape(means,
                       (-1,) + self.input_shape)  # [n_components, input_shape]
    qz = self.fn_encoding(means)  # [n_components, zdim]
    return qz._sample_n(1, seed=seed)


class VampriorVAE(BetaVAE):
  """ An example of Vamprior VAE. In practice, vamprior could be integrated into
  any VAE model by changing the prior of specified latents variable.

  References
  ----------
  Tomczak, J.M., Welling, M., 2018. VAE with a VampPrior. arXiv:1705.07120 [cs, stat].

  """

  def __init__(
      self,
      n_components: int = 500,
      pseudoinputs_mean: float = -0.05,
      pseudoinputs_std: float = 0.01,
      beta: Union[float, Interpolation] = linear(vmin=1e-6,
                                                 vmax=1.,
                                                 steps=2000,
                                                 delay_in=0),
      **kwargs):
    super().__init__(beta=beta, **kwargs)
    self.n_components = n_components
    self.pseudoinputs_mean = pseudoinputs_mean
    self.pseudoinputs_std = pseudoinputs_std

  def build(self, input_shape=None):
    ret = super().build(input_shape)
    self.vamprior = Vamprior(input_shape=input_shape[1:],
                             fn_encoding=self.encode,
                             n_components=self.n_components,
                             pseudoinputs_mean=self.pseudoinputs_mean,
                             pseudoinputs_std=self.pseudoinputs_std,
                             dtype=self.dtype)
    self.latents.prior = self.vamprior
    return ret
