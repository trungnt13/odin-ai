from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors.exp import Exp
from tensorflow_probability.python.distributions import (
    LogNormal, TransformedDistribution, Uniform)
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    "LogUniform",
]


class LogUniform(TransformedDistribution):
  """The log-uniform distribution (i.e. the logarithm of the
  samples from this distribution are Uniform) """

  def __init__(self,
               low=0.,
               high=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="LogUniform"):
    """Construct a log-normal distribution.

    The LogNormal distribution models positive-valued random variables
    whose logarithm is normally distributed with mean `loc` and
    standard deviation `scale`. It is constructed as the exponential
    transformation of a Normal distribution.

    Args:
      low: Floating point tensor, lower boundary of the output interval. Must
        have `low < high`.
      high: Floating point tensor, upper boundary of the output interval. Must
        have `low < high`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([low, high], tf.float32)
      super(LogUniform, self).__init__(distribution=Uniform(
          low=tf.convert_to_tensor(value=low, name="low", dtype=dtype),
          high=tf.convert_to_tensor(value=high, name="high", dtype=dtype),
          allow_nan_stats=allow_nan_stats),
                                       bijector=Exp(),
                                       validate_args=validate_args,
                                       parameters=parameters,
                                       name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("low", "high"),
            ([tf.convert_to_tensor(value=sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(low=0, high=0)

  @property
  def low(self):
    """Lower boundary of the output interval."""
    return self.distribution.low

  @property
  def high(self):
    """Upper boundary of the output interval."""
    return self.distribution.high

  def range(self, name="range"):
    """`high - low`."""
    with self._name_scope(name):
      return self.high - self.low

  def _entropy(self):
    raise NotImplementedError

  def _mean(self):
    raise NotImplementedError

  def _variance(self):
    raise NotImplementedError

  def _stddev(self):
    raise NotImplementedError
