from __future__ import absolute_import, division, print_function

import collections

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.layers.distribution_layer import (
    DistributionLambda, _get_convert_to_tensor_fn, _serialize)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible as dtc
from tensorflow_probability.python.layers.internal import \
    tensor_tuple as tensor_tuple

__all__ = [
    'ConcatDistribution', 'Sampling', 'Moments', 'Stddev', 'DistributionAttr'
]


def _check_distribution(x):
  assert isinstance(x, Distribution), \
  "Input to this layer must be instance of tensorflow_probability Distribution"


class ConcatDistribution(DistributionLambda):
  """ This layer create a new `Distribution` by concatenate parameters of
  multiple distributions of the same type along given `axis`
  """

  def __init__(self,
               axis=None,
               convert_to_tensor_fn=Distribution.sample,
               **kwargs):
    from odin.bay.distributions.utils import concat_distribution
    super(ConcatDistribution, self).__init__(
        lambda dists: concat_distribution(dists=dists, axis=axis),
        convert_to_tensor_fn, **kwargs)
    self.axis = axis

  def get_config(self):
    config = super(ConcatDistribution, self).get_config()
    config['axis'] = self.axis
    return config


class Sampling(Layer):
  """ Sample the output from tensorflow-probability
  distribution layers """

  def __init__(self, n_samples=None, **kwargs):
    super(Sampling, self).__init__(**kwargs)
    self.n_samples = n_samples

  def get_config(self):
    config = super(Sampling, self).get_config()
    config['n_samples'] = self.n_samples
    return config

  def call(self, x, n_samples=None, **kwargs):
    if not isinstance(x, Distribution):
      return tf.expand_dims(x, axis=0)

    if n_samples is None:
      n_samples = self.n_samples
    return x.sample() if n_samples is None else x.sample(n_samples)


class Moments(Layer):
  """ Moments """

  def __init__(self, mean=True, variance=True, **kwargs):
    super(Moments, self).__init__(**kwargs)
    self.mean = bool(mean)
    self.variance = bool(variance)
    assert self.mean or self.variance, "This layer must return mean or variance"

  def call(self, x, **kwargs):
    if not isinstance(x, Distribution):
      return x

    outputs = []
    if self.mean:
      outputs.append(x.mean())
    if self.variance:
      outputs.append(x.variance())
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

  def get_config(self):
    config = super(Moments, self).get_config()
    config['mean'] = self.mean
    config['variance'] = self.variance
    return config

  def compute_output_shape(self, input_shape):
    return [input_shape, input_shape] \
      if self.mean and self.variance else input_shape


class Stddev(Layer):
  """ Get standard deviation of an input distribution, return identity
  if the input is not an instance of `Distribution` """

  def __init__(self, **kwargs):
    super(Stddev, self).__init__(**kwargs)

  def call(self, x, **kwargs):
    if not isinstance(x, Distribution):
      return x

    return x.stddev()

  def compute_output_shape(self, input_shape):
    return input_shape


class DistributionAttr(Layer):
  """ This layer provide convenient way to extract statistics stored
  as attributes of the `Distribution` """

  def __init__(self,
               attr_name,
               convert_to_tensor_fn=Distribution.sample,
               **kwargs):
    super(DistributionAttr, self).__init__(**kwargs)
    self.attr_name = str(attr_name)
    if isinstance(convert_to_tensor_fn, property):
      convert_to_tensor_fn = convert_to_tensor_fn.fget
    self.convert_to_tensor_fn = convert_to_tensor_fn

  def get_config(self):
    config = super(DistributionAttr, self).get_config()
    config['attr_name'] = self.attr_name
    config['convert_to_tensor_fn'] = self.convert_to_tensor_fn
    return config

  def call(self, x):
    attrs = self.attr_name.split('.')
    for a in attrs:
      x = getattr(x, a)
    # special case a distribution is returned
    if isinstance(x, Distribution) and not isinstance(x, dtc._TensorCoercible):
      dist = dtc._TensorCoercible(
          distribution=x, convert_to_tensor_fn=self.convert_to_tensor_fn)
      value = tf.convert_to_tensor(value=dist)
      value._tfp_distribution = dist
      dist.shape = value.shape
      dist.get_shape = value.get_shape
      x = dist
    return x
