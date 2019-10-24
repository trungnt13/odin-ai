# -*- coding: utf-8 -*-
# This module contain the implementation of Time-delayed neural network
# based on variate source, using both Dense (classic) and Convolution (modern)
# for modeling delayed inputs.
#
# The inputs could be 3-D or 4-D tensor
#  [n_samples, n_timestep, n_features]
# or
#  [n_samples, n_timestep, n_features, n_channels]
#
# References
# ----------
# [1] Waibel, A., Hanazawa, T., Hinton, G., Shikano, K., & Lang, K. J. (1989).
# Phoneme recognition using time-delay neural networks. IEEE transactions on
# acoustics, speech, and signal processing, 37(3), 328-339.
# [2] Peddinti, V., Povey, D., & Khudanpur, S. (2015). A time delay neural
# network architecture for efficient modeling of long temporal contexts.
# In INTERSPEECH (pp. 3214-3218).
# [3] Yoon Kim, Yacine Jernite, David Sontag, and Alexander M. Rush. 2016.
# Character-aware neural language models, AAAI'16
from __future__ import absolute_import, division, print_function

import sys
import types as python_types

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python.keras import Model, activations
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.layers import Conv1D, Dense, Layer, LeakyReLU
from tensorflow.python.keras.utils import generic_utils

from odin.backend import parse_reduction
from odin.utils import as_tuple

__all__ = ['TimeDelay', 'TimeDelayDense', 'TimeDelayConv', 'TimeDelayConvTied']


class TimeDelay(Model):
  """ A generalized implementation of time-delayed neural network by applying

  Parameters
  ----------
  fn_layer_creator : `callable`
    a function that returns a `keras.Layer`
  delay_context : list of `int`
    list of time delay taken into account
  pooling : {'none', 'sum', 'min', 'max', 'avg', 'stat'} (default='sum')
    pooling in time dimension after convolution operator
    for 'stat' pooling, mean and standard deviation is calculated along
    time-dimension, then output the concatenation of the two.
    if None, no pooling is performed, the output is returned in
    shape `[n_samples, n_reduced_timestep, n_new_features]`

  Input shape
  -----------
    3D tensor with shape: `(batch_size, timesteps, input_dim)`

  Output shape
  ------------
    3D tensor with shape: `(batch_size, new_timesteps, units)`

  """

  def __init__(self,
               fn_layer_creator,
               delay_context=(-2, -1, 0, 1, 2),
               pooling='sum',
               name=None,
               **kwargs):
    super(TimeDelay, self).__init__(name=name, **kwargs)
    assert callable(fn_layer_creator), \
      "fn_layer_creator must be callable and return a keras.Layer"

    self.fn_layer_creator = fn_layer_creator
    # no duplicated frame index
    self.delay_context = np.array(sorted(set(int(i) for i in delay_context)))
    self.context_length = self.delay_context[-1] - self.delay_context[0] + 1

    self.delays = self.delay_context + max(0, -self.delay_context[0])
    self.min_delay = max(0, min(self.delays))

    # pooling function for aggrevate the time outputs
    self.pooling = 'none' if pooling is None else pooling
    self.fn_pooling = parse_reduction(pooling)

    all_layers = []
    for time_id in range(len(self.delay_context)):
      layer = fn_layer_creator()
      assert isinstance(layer, Layer), \
        "fn_layer_creator must return keras.Layer instance, but return type is %s" % \
          str(type(layer))
      # we need to setattr so the Model will manage the Layer
      setattr(self, 'time_layer_%d' % time_id, layer)
      all_layers.append(layer)

    assert len(self.layers) == len(self.delay_context), \
      "Number of layers and length of time context mismatch!"
    self.all_layers = all_layers

  def call(self, inputs, training=None):
    # anyway, if the smallest value is negative,
    # start from 0 (i.e. relative position)
    shape = tf.shape(inputs)
    timestep = shape[1]
    y = []

    for delay, layer in zip(self.delays, self.all_layers):
      start = delay
      end = timestep - self.context_length + delay + 1 - self.min_delay
      y.append(tf.expand_dims(layer(inputs[:, start:end]), axis=0))

    y = tf.concat(y, axis=0)
    y = self.fn_pooling(y, axis=0)

    if isinstance(self.pooling, string_types) and \
      'none' in self.pooling.lower() and \
        self.context_length == 1:
      y = tf.squeeze(y, axis=0)
    return y

  @classmethod
  def from_config(cls, config, custom_objects=None):
    fn, fn_type, fn_module = config['fn_layer_creator']

    globs = globals()
    module = config.pop(fn_module, None)
    if module in sys.modules:
      globs.update(sys.modules[module].__dict__)
    if custom_objects:
      globs.update(custom_objects)

    if fn_type == 'function':
      # Simple lookup in custom objects
      fn = generic_utils.deserialize_keras_object(
          fn,
          custom_objects=custom_objects,
          printable_module_name='function in Lambda layer')
    elif fn_type == 'lambda':
      # Unsafe deserialization from bytecode
      fn = generic_utils.func_load(fn, globs=globs)
    config['fn_layer_creator'] = fn

    return cls(**config)

  def get_config(self):
    configs = super(Network, self).get_config()

    fn = self.fn_layer_creator
    if isinstance(fn, python_types.LambdaType):
      fn = (generic_utils.func_dump(fn), 'lambda', fn.__module__)
    elif callable(fn):
      fn = (fn.__name__, 'function', fn.__module__)

    configs.update({
        'fn_layer_creator': fn,
        'delay_context': self.delay_context,
        'pooling': self.pooling,
    })
    return configs


class TimeDelayDense(TimeDelay):
  """ The implementaiton of time delay neural network

  Input shape
  -----------
    3D tensor with shape: `(batch_size, timesteps, input_dim)`

  Output shape
  ------------
    3D tensor with shape: `(batch_size, new_timesteps, units)`

  """

  def __init__(self,
               units,
               delay_context=(-2, -1, 0, 1, 2),
               pooling='sum',
               activation='linear',
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(TimeDelayDense, self).__init__(fn_layer_creator=lambda: Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    ),
                                         delay_context=delay_context,
                                         pooling=pooling,
                                         **kwargs)


class TimeDelayConv(TimeDelay):
  """ This implementaiton create multiple convolutional neural network for
  each time delay.

  Parameters
  ----------

  Input shape
  -----------
    3D tensor with shape: `(batch_size, timesteps, input_dim)`

  Output shape
  ------------
    3D tensor with shape: `(batch_size, new_timesteps, units)`
      `steps` value might have changed due to padding or strides.

  """

  def __init__(self,
               units,
               kernel_size=3,
               delay_context=(-2, -1, 0, 1, 2),
               pooling='sum',
               activation='linear',
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(TimeDelayConv, self).__init__(fn_layer_creator=lambda: Conv1D(
        filters=units,
        kernel_size=kernel_size,
        strides=1,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    ),
                                        delay_context=delay_context,
                                        pooling=pooling,
                                        **kwargs)


class TimeDelayConvTied(TimeDelay):
  """ Time-delayed dense implementation but using a 1D-convolutional
  neural network, only support continuos delay context (given a number
  of `delay_strides`).

  From the paper, it is suggested to create multiple `TimeDelayedConv`
  with variate number of feature map and length of context windows,
  then concatenate the outputs for `Dense` layers

  For example:
   - feature_maps = [50, 100, 150, 200, 200, 200, 200]
   - kernels = [1, 2, 3, 4, 5, 6, 7]

  Parameters
  ----------
  units : `int`
    number of new features
  delay_length : `int` (default=5)
    length of time delayed context
  delay_strides : `int` (default=1)
    specifying the strides of time window

  """

  def __init__(self,
               units,
               delay_length=5,
               delay_strides=1,
               activation='linear',
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(TimeDelayConvTied, self).__init__(fn_layer_creator=lambda: Conv1D(
        filters=units,
        kernel_size=delay_length,
        strides=delay_strides,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    ),
                                            delay_context=(0,),
                                            pooling='none',
                                            **kwargs)
