from __future__ import print_function, division, absolute_import

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc

class ReduceMean(Layer):
  """ ReduceMean """

  def __init__(self, axis=None, keepdims=None, **kwargs):
    super(ReduceMean, self).__init__(**kwargs)
    self.axis = axis
    self.keepdims = keepdims

  def call(self, x):
    return [tf.reduce_mean(i, axis=self.axis, keepdims=self.keepdims) for i in x] \
    if isinstance(x, (tuple, list)) else \
    tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)

class Moments(Layer):
  """ Moments """

  def __init__(self, mean=True, variance=True, **kwargs):
    super(Moments, self).__init__(**kwargs)
    self.mean = bool(mean)
    self.variance = bool(variance)
    assert self.mean or self.variance, "This layer must return mean or variance"

  def call(self, x):
    assert isinstance(x, Distribution), \
    "Input to this layer must be instance of tensorflow_probability Distribution"
    outputs = []
    if self.mean:
      outputs.append(x.mean())
    if self.variance:
      outputs.append(x.variance())
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

  def compute_output_shape(self, input_shape):
    return [input_shape, input_shape] if self.mean and self.variance else input_shape

class Stddev(Layer):

  def __init__(self, **kwargs):
    super(Stddev, self).__init__(**kwargs)

  def call(self, x):
    assert isinstance(x, Distribution), \
    "Input to this layer must be instance of tensorflow_probability Distribution"
    return x.stddev()

  def compute_output_shape(self, input_shape):
    return input_shape

class GetAttr(Layer):
  """ GetAttr """

  def __init__(self, attr_name, convert_to_tensor_fn=Distribution.sample, **kwargs):
    super(GetAttr, self).__init__(**kwargs)
    self.attr_name = str(attr_name)
    if isinstance(convert_to_tensor_fn, property):
      convert_to_tensor_fn = convert_to_tensor_fn.fget
    self.convert_to_tensor_fn = convert_to_tensor_fn

  def call(self, x):
    attrs = self.attr_name.split('.')
    for a in attrs:
      x = getattr(x, a)
    # special case a distribution is returned
    if isinstance(x, Distribution) and not isinstance(x, dtc._TensorCoercible):
      dist = dtc._TensorCoercible(
          distribution=x,
          convert_to_tensor_fn=self.convert_to_tensor_fn)
      value = tf.convert_to_tensor(value=dist)
      value._tfp_distribution = dist
      dist.shape = value.shape
      dist.get_shape = value.get_shape
      x = dist
    return x
