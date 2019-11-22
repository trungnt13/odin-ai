from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import Distribution

__all__ = ['Reduce', 'LogNorm']


class LogNorm(keras.layers.Layer):

  def call(self, x, **kwargs):
    return tf.math.log1p(x)


class Reduce(keras.layers.Layer):
  r""" Customized reduction with Lambda
  Arguments:
    op : String. One of the following: 'mean', 'sum', 'prod', 'max', 'min',
      'logsumexp', 'any', 'all'
  """

  def __init__(self, op, axis=None, keepdims=None):
    if not callable(op):
      op = str(op).lower()
      if op == 'mean':
        op = tf.reduce_mean
      elif op == 'sum':
        op = tf.reduce_sum
      elif op == 'prod':
        op = tf.reduce_prod
      elif op == 'max':
        op = tf.reduce_max
      elif op == 'min':
        op = tf.reduce_min
      elif op == 'logsumexp':
        op = tf.reduce_logsumexp
      elif op == 'any':
        op = tf.reduce_any
      elif op == 'all':
        op = tf.reduce_all
    else:
      args = inspect.getfullargspec(op)
      assert 'axis' in args and 'keepdims' in args, \
        "reduce function must has 2 arguments: 'axis' and 'keepdims'"
    super(Reduce, self).__init__(name=op.__name__)
    self.op = op
    self.axis = axis
    self.keepdims = keepdims

  def get_config(self):
    config = super(Reduce, self).get_config()
    config['op'] = self.op
    config['axis'] = self.axis
    config['keepdims'] = self.keepdims
    return config

  def call(self, x, **kwargs):
    if isinstance(x, (tuple, list)):
      return [self.op(i, axis=self.axis, keepdims=self.keepdims) for i in x]
    return self.op(x, axis=self.axis, keepdims=self.keepdims)
