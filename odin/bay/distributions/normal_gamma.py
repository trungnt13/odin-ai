"""The NormalGamma distribution class."""
from __future__ import absolute_import, division, print_function

import math
import functools
# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops

__all__ = [
    "NormalGamma",
]

class NormalGamma(distribution.Distribution):
  """Normal-Gamma distribution.
    [normal gamma](https://en.wikipedia.org/wiki/Normal-gamma_distribution)

  The normal-gamma distribution (or Gaussian-gamma distribution)
  is a bivariate four-parameter family of continuous probability
  distributions.

  It is the conjugate prior of a normal distribution with
  unknown mean and precision.

  """

  def __init__(self,
               loc,
               scale,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="NormalGamma"):
    """Initializes a batch of Normal-Gamma distributions.



    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
        (`scale = sqrt(lambda)` according to the wikipedia article)
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(
        name, values=[loc, scale, concentration, rate]):
      dtype = dtype_util.common_dtype(
          [loc, scale, concentration, rate],
          preferred_dtype=tf.float32)
      loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      scale = tf.convert_to_tensor(
          scale, name="scale", dtype=dtype)
      concentration = tf.convert_to_tensor(
          concentration, name="concentration", dtype=dtype)
      rate = tf.convert_to_tensor(rate, name="rate", dtype=dtype)

      with tf.control_dependencies([
          tf.assert_positive(scale),
          tf.assert_positive(concentration),
          tf.assert_positive(rate),
      ] if validate_args else []):
        self._loc = tf.identity(loc)
        self._scale = tf.identity(scale)
        self._concentration = tf.identity(concentration)
        self._rate = tf.identity(rate)
        tf.assert_same_float_dtype(
            [self._loc, self._scale,
             self._concentration, self._rate])
      # the coefficient for the precision
      self._lambda = tf.square(self._scale)
    super(NormalGamma, self).__init__(
        dtype=self._loc.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._loc, self._scale,
                       self._concentration, self._rate],
        name=name)

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the standard deviation."""
    return self._scale

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  # ******************** Shape and sampling ******************** #

  def _batch_shape_tensor(self):
    tensors = [self.loc, self.scale,
               self.concentration, self.rate]
    return functools.reduce(tf.broadcast_dynamic_shape,
                            [tf.shape(tensor) for tensor in tensors])

  def _batch_shape(self):
    tensors = [self.loc, self.scale,
               self.concentration, self.rate]
    return functools.reduce(tf.broadcast_static_shape,
                            [tensor.shape for tensor in tensors])

  def _event_shape_tensor(self):
    return tf.constant([2], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.vector(2)

  def _sample_n(self, n, seed=None):
    seed = seed_stream.SeedStream(seed, "normal_gamma")
    shape = tf.concat([[n], self.batch_shape_tensor()], 0)

    precision = tf.random_gamma(
        shape=shape,
        alpha=self.concentration,
        beta=self.rate,
        dtype=self.dtype,
        seed=seed())

    scale = tf.sqrt(1 / (self._lambda * precision))
    mean = tf.random_normal(
        shape=shape, mean=0., stddev=1., dtype=self.loc.dtype, seed=seed())
    mean = mean * scale + self.loc

    return tf.concat((tf.expand_dims(mean, axis=-1),
                      tf.expand_dims(precision, axis=-1)), axis=-1)

  # ******************** Log probability ******************** #

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    mean = tf.squeeze(tf.gather(x, [0], axis=-1), axis=-1)
    precision = self._maybe_assert_valid_sample(
        tf.squeeze(tf.gather(x, [1], axis=-1), axis=-1))
    return (tf.math.xlogy(self.concentration - 0.5, precision)
            - self.rate * precision
            - 0.5 * self._lambda * precision * tf.square(mean - self.loc))

  def _log_normalization(self):
    return (tf.lgamma(self.concentration) + 0.5 * math.log(2. * math.pi)
            - self.concentration * tf.log(self.rate)
            - 0.5 * tf.log(self._lambda))

  # ******************** Moments ******************** #

  def _mean(self):
    mean = self.loc * tf.ones_like(self.scale)
    precision = (self.concentration / self.rate) * tf.ones_like(mean)
    return tf.concat((tf.expand_dims(mean, axis=-1),
                      tf.expand_dims(precision, axis=-1)), axis=-1)

  def _maybe_assert_valid_sample(self, x):
    tf.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        tf.assert_positive(x),
    ], x)
