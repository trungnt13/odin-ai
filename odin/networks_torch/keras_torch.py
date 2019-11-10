from __future__ import absolute_import, division, print_function

import inspect
import os

import numpy as np
import torch
from six import add_metaclass
from tensorflow.python.keras.utils import conv_utils
from torch import nn
from torch.nn import functional

from odin.backend import (parse_activation, parse_constraint, parse_initializer,
                          parse_regularizer)


class Lambda(nn.Module):

  def __init__(self, function):
    super().__init__()
    self._function = function

  def forward(self, *args, **kwargs):
    return self._function(*args, **kwargs)


class Layer(nn.Module):

  def __init__(self, **kwargs):
    super().__init__()
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
    # call
    inputs = inputs[0] if n_inputs == 1 else inputs
    # this make life easier but not the solution for everything
    if isinstance(inputs, np.ndarray):
      inputs = torch.Tensor(inputs)
    # intelligent call
    specs = inspect.getfullargspec(self.call)
    if specs.varkw is not None:
      return self.call(inputs, **kwargs)
    kw = {i: kwargs[i] for i in specs.args[2:] if i in kwargs}
    return self.call(inputs, **kw)

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
    D_in = input_shape[-1]
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


# ===========================================================================
# Recurrent neural network
# ===========================================================================
class _RNNLayer(Layer):

  def __init__(self, kernel_initializer, recurrent_initializer,
               bias_initializer, return_sequences, return_state, go_backwards,
               stateful, **kwargs):
    super(_RNNLayer, self).__init__(**kwargs)
    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    if stateful:
      raise NotImplementedError(
          "pytorch currently does not support stateful RNN")
    self.kernel_initializer = parse_initializer(kernel_initializer, self)
    self.recurrent_initializer = parse_initializer(recurrent_initializer, self)
    self.bias_initializer = parse_initializer(bias_initializer, self)

  def build(self, input_shape):
    if not hasattr(self, '_rnn'):
      raise RuntimeError(
          "instance of pytorch RNN must be create and assigned to attribute "
          "name '_rnn' during `build`.")

    for layer_idx in range(self._rnn.num_layers):
      self.kernel_initializer(getattr(self._rnn, 'weight_ih_l%d' % layer_idx))
      self.recurrent_initializer(getattr(self._rnn,
                                         'weight_hh_l%d' % layer_idx))
      b_ih = getattr(self._rnn, 'bias_ih_l%d' % layer_idx)
      b_hh = getattr(self._rnn, 'bias_hh_l%d' % layer_idx)
      self.bias_initializer(b_ih)
      self.bias_initializer(b_hh)
      if getattr(self, 'unit_forget_bias', False):
        # b_ii|b_if|b_ig|b_io
        b_ih[self.units:self.units * 2] = 1
        # b_hi|b_hf|b_hg|b_ho
        b_hh[self.units:self.units * 2] = 1
    return super(_RNNLayer, self).build(input_shape)

  def call(self, inputs):
    if self.go_backwards:
      inputs = inputs.flip(1)

    outputs, states = self._rnn(inputs)
    if not isinstance(states, (tuple, list)):
      states = (states,)

    if not self.return_sequences:
      outputs = outputs[:, -1]
    if not self.return_state:
      return outputs

    return [outputs] + list(states)


class SimpleRNN(_RNNLayer):

  def __init__(self,
               units,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               num_layers=1,
               bidirectional=False,
               **kwargs):
    super(SimpleRNN, self).__init__(kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    **kwargs)
    self.units = int(units)
    self.activation = str(activation)
    self.use_bias = use_bias
    self.dropout = dropout
    self.num_layers = num_layers
    self.bidirectional = bidirectional

  def build(self, input_shape):
    input_size = input_shape[-1]
    self._rnn = nn.RNN(input_size=input_size,
                       hidden_size=self.units,
                       num_layers=self.num_layers,
                       nonlinearity=self.activation,
                       bias=self.use_bias,
                       batch_first=True,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional)
    return super(SimpleRNN, self).build(input_shape)


class LSTM(_RNNLayer):

  def __init__(self,
               units,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               num_layers=1,
               bidirectional=False,
               **kwargs):
    super(LSTM, self).__init__(kernel_initializer=kernel_initializer,
                               recurrent_initializer=recurrent_initializer,
                               bias_initializer=bias_initializer,
                               return_sequences=return_sequences,
                               return_state=return_state,
                               go_backwards=go_backwards,
                               stateful=stateful,
                               **kwargs)
    self.units = int(units)
    self.use_bias = use_bias
    self.dropout = dropout
    self.unit_forget_bias = unit_forget_bias
    self.num_layers = num_layers
    self.bidirectional = bidirectional

  def build(self, input_shape):
    input_size = input_shape[-1]
    self._rnn = nn.LSTM(input_size=input_size,
                        hidden_size=self.units,
                        num_layers=self.num_layers,
                        bias=self.use_bias,
                        batch_first=True,
                        dropout=self.dropout,
                        bidirectional=self.bidirectional)
    return super(LSTM, self).build(input_shape)


class GRU(_RNNLayer):

  def __init__(self,
               units,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               num_layers=1,
               bidirectional=False,
               **kwargs):
    super(GRU, self).__init__(kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              return_sequences=return_sequences,
                              return_state=return_state,
                              go_backwards=go_backwards,
                              stateful=stateful,
                              **kwargs)
    self.units = int(units)
    self.use_bias = use_bias
    self.dropout = dropout
    self.num_layers = num_layers
    self.bidirectional = bidirectional

  def build(self, input_shape):
    input_size = input_shape[-1]
    self._rnn = nn.GRU(input_size=input_size,
                       hidden_size=self.units,
                       num_layers=self.num_layers,
                       bias=self.use_bias,
                       batch_first=True,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional)
    return super(GRU, self).build(input_shape)
