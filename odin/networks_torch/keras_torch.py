from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from six import add_metaclass
from tensorflow.python.keras.utils import conv_utils
from torch import nn
from torch.nn import functional

from odin.backend import (parse_activation, parse_constraint, parse_initializer,
                          parse_regularizer)


class Layer(nn.Module):

  def __init__(self):
    super(Layer, self).__init__()
    self.built = False

  def build(self, input_shape):
    """Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Arguments:
      input_shape: Instance of `TensorShape`, or list of instances of
        `TensorShape` if the layer expects a list of inputs
        (one instance per input).
    """
    self.built = True

  def forward(self, *inputs, **kwargs):
    n_inputs = len(inputs)
    input_shape = [i.shape for i in inputs]
    if n_inputs == 1:
      input_shape = input_shape[0]
    if not self.built:
      self.build(input_shape)
    return self.call(inputs[0] if n_inputs == 1 else inputs, **kwargs)

  def call(self, inputs, **kwargs):
    raise NotImplementedError


class Dense(Layer):

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(Dense, self).__init__()
    self.units = int(units)
    self.activation = parse_activation(activation, self)
    self.use_bias = use_bias
    self.kernel_initializer = parse_initializer(kernel_initializer, self)
    self.bias_initializer = parse_initializer(bias_initializer, self)

  def build(self, input_shape):
    D_in = input_shape[1]
    D_out = self.units
    self._linear = nn.Linear(in_features=D_in,
                             out_features=D_out,
                             bias=self.use_bias)
    self.kernel_initializer(self._linear.weight)
    if self.use_bias:
      self.bias_initializer(self._linear.bias)
    return super(Dense, self).build(input_shape)

  def call(self, inputs, training=None):
    y = self._linear(inputs)
    return self.activation(y)


# ===========================================================================
# Convolution
# ===========================================================================
class Conv(Layer):

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               name=None,
               **kwargs):
    super(Conv, self).__init__()
    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                  'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if (self.padding == 'causal' and not isinstance(self, (Conv1D,))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and ``SeparableConv1D`.')
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                    'dilation_rate')
    self.activation = parse_activation(activation, self)
    self.use_bias = use_bias
    self.kernel_initializer = parse_initializer(kernel_initializer, self)
    self.bias_initializer = parse_initializer(bias_initializer, self)

  def build(self, input_shape):
    in_channels = (input_shape[-1]
                   if self.data_format == 'channels_last' else input_shape[1])
    spatial_shape = (input_shape[2:] if self.data_format == 'channels_first'
                     else input_shape[1:-1])
    # 1D `(padW,)`
    # 2D `(padH, padW)`
    # 3D `(padT, padH, padW)`
    if self.padding == 'valid':
      padding = 0
      padding_mode = 'zeros'
    elif self.padding == 'same':
      padding = [i // 2 for i in self.kernel_size]
      padding_mode = 'zeros'
    elif self.padding == 'causal' and self.rank == 1:
      padding = 0
      padding_mode = 'zeros'
    else:
      raise NotImplementedError("No support for padding='%s' and rank=%d" %
                                (self.padding, self.rank))

    if self.rank == 1:
      self._conv = torch.nn.Conv1d(in_channels=in_channels,
                                   out_channels=self.filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.strides,
                                   padding=padding,
                                   dilation=self.dilation_rate,
                                   groups=1,
                                   bias=self.use_bias,
                                   padding_mode=padding_mode)
    elif self.rank == 2:
      self._conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=self.filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.strides,
                                   padding=padding,
                                   dilation=self.dilation_rate,
                                   groups=1,
                                   bias=self.use_bias,
                                   padding_mode=padding_mode)
    elif self.rank == 3:
      self._conv = torch.nn.Conv3d(in_channels=in_channels,
                                   out_channels=self.filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.strides,
                                   padding=padding,
                                   dilation=self.dilation_rate,
                                   groups=1,
                                   bias=True,
                                   padding_mode=padding_mode)
    else:
      raise NotImplementedError("No support for rank=%d" % self.rank)

    self.kernel_initializer(self._conv.weight)
    if self.use_bias:
      self.bias_initializer(self._conv.bias)
    return super(Conv, self).build(input_shape)

  def call(self, inputs, training=None):
    # causal padding for temporal signal
    if self.padding == 'causal' and self.rank == 1:
      inputs = functional.pad(inputs,
                              self._compute_causal_padding(),
                              mode='constant',
                              value=0)
    # pytorch only support channels_first
    if self.data_format == 'channels_last':
      inputs = inputs.transpose(1, -1)
    # applying the convolution
    y = self._conv(inputs)
    if self.data_format == 'channels_last':
      y = y.transpose(1, -1)
    return y

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers.
    @Original code: tensorflow.keras
    """
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [0, 0, left_pad, 0, 0, 0]
    else:
      causal_padding = [0, 0, 0, 0, left_pad, 0]
    return causal_padding


class Conv1D(Conv):

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(Conv1D, self).__init__(rank=1,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 data_format=data_format,
                                 dilation_rate=dilation_rate,
                                 activation=activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 **kwargs)


class ConvCausal(Conv1D):

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(ConvCausal, self).__init__(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding='causal',
                                     data_format=data_format,
                                     dilation_rate=dilation_rate,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     **kwargs)


class Conv2D(Conv):

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(Conv2D, self).__init__(rank=2,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 data_format=data_format,
                                 dilation_rate=dilation_rate,
                                 activation=activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 **kwargs)


class Conv3D(Conv):

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(Conv3D, self).__init__(rank=3,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 data_format=data_format,
                                 dilation_rate=dilation_rate,
                                 activation=activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 **kwargs)
