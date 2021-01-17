from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from odin.backend.interpolation import Interpolation, linear
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.utils import as_tuple
from tensorflow import keras
from tensorflow_probability.python.distributions import (NOT_REPARAMETERIZED,
                                                         Distribution, Normal)

__all__ = ['Vamprior', 'vampriorVAE']


def hard_probs(x):
  return tf.clip_by_value(x, 0.0, 1.0)


class Vamprior(Distribution):

  def __init__(self,
               input_shape: List[int],
               fn_encoding: Callable[[tf.Tensor], Distribution],
               n_components: int = 500,
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


class vampriorVAE(betaVAE):

  def __init__(self,
               n_components: int = 500,
               pseudoinputs_mean: float = -0.05,
               pseudoinputs_std: float = 0.01,
               beta: Union[float, Interpolation] = linear(vmin=1e-6,
                                                          vmax=1.,
                                                          length=2000,
                                                          delay_in=0),
               name: str = 'VampriorVAE',
               **kwargs):
    super().__init__(beta=beta, name=name, **kwargs)
    self.n_components = n_components
    self.pseudoinputs_mean = pseudoinputs_mean
    self.pseudoinputs_std = pseudoinputs_std

  def build(self, input_shape):
    ret = super().build(input_shape)
    self.vamprior = Vamprior(input_shape=input_shape[1:],
                             fn_encoding=self.encode,
                             n_components=self.n_components,
                             pseudoinputs_mean=self.pseudoinputs_mean,
                             pseudoinputs_std=self.pseudoinputs_std,
                             dtype=self.dtype)
    self.latents.prior = self.vamprior
    return ret
