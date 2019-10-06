from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from six import string_types
from torch import nn

from odin.backend import concatenate, expand_dims, parse_reduction, squeeze
from odin.networks_torch.keras_torch import Conv1D, Dense, Layer
from odin.utils import as_tuple


class TimeDelay(Layer):
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
               **kwargs):
    super(TimeDelay, self).__init__(**kwargs)
    assert callable(fn_layer_creator), \
      "fn_layer_creator must be callable and return a torch.nn.Module"

    self.fn_layer_creator = fn_layer_creator
    # no duplicated frame index
    self.delay_context = np.array(sorted(set(int(i) for i in delay_context)))
    self.context_length = self.delay_context[-1] - self.delay_context[0] + 1

    self.delays = self.delay_context + max(0, -self.delay_context[0])
    self.min_delay = max(0, min(self.delays))

    # pooling function for aggrevate the time outputs
    self.pooling = 'none' if pooling is None else pooling
    self.fn_pooling = parse_reduction(pooling)

    all_layers = nn.ModuleList()
    for time_id in range(len(self.delay_context)):
      layer = fn_layer_creator()
      assert isinstance(layer, torch.nn.Module), \
        "fn_layer_creator must return torch.nn.Module instance, " + \
          "but return type is %s" % \
          str(type(layer))
      # we need to setattr so the Model will manage the Layer
      all_layers.append(layer)
    self.all_layers = all_layers

  def call(self, inputs, training=None):
    # anyway, if the smallest value is negative,
    # start from 0 (i.e. relative position)
    shape = inputs.shape
    timestep = shape[1]
    y = []

    for delay, layer in zip(self.delays, self.all_layers):
      start = delay
      end = timestep - self.context_length + delay + 1 - self.min_delay
      y.append(expand_dims(layer(inputs[:, start:end]), axis=0))

    y = concatenate(y, axis=0)
    y = self.fn_pooling(y, axis=0)

    if isinstance(self.pooling, string_types) and \
      'none' in self.pooling.lower() and \
        self.context_length == 1:
      y = squeeze(y, axis=0)
    return y


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
               **kwargs):
    super(TimeDelayDense, self).__init__(fn_layer_creator=lambda: Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
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
    ),
                                        delay_context=delay_context,
                                        pooling=pooling,
                                        **kwargs)


class TimeDelayConvTied(TimeDelay):
  """ Time-delayed dense implementation but using a 1D-convolutional
  neural network, only support consecutive delay context (given a number
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
    ),
                                            delay_context=(0,),
                                            pooling='none',
                                            **kwargs)
