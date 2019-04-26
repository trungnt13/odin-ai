from __future__ import absolute_import, division, print_function

import collections
import math

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import (RNNCell, MultiRNNCell,
    DropoutWrapper, EmbeddingWrapper,
    InputProjectionWrapper, OutputProjectionWrapper,
    PhasedLSTMCell, GridLSTMCell, GLSTMCell,
    IntersectionRNNCell, UGRNNCell, NASCell,
    LayerNormBasicLSTMCell, HighwayWrapper,
    AttentionCellWrapper, BidirectionalGridLSTMCell,
    TimeFreqLSTMCell, CoupledInputForgetGateLSTMCell)
from tensorflow.python.ops.rnn_cell_impl import (GRUCell, LSTMCell, BasicLSTMCell)
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from odin.backend.helpers import get_all_variables, get_value
from odin.autoconfig import get_session


class BasicRNNCell(RNNCell):
  """The most basic RNN cell.

  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units,
               kernel_initializer=None,
               bias_initializer=tf.constant_initializer(value=0.),
               activation=None,
               reuse=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.nn.tanh
    self._bias_initializer = bias_initializer
    self._kernel_initializer = kernel_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    from tensorflow.python.ops.rnn_cell_impl import _linear
    output = self._activation(
        _linear([inputs, state], output_size=self._num_units,
        bias=False if self._bias_initializer is None else True,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer))
    return output, output


class AttentionCell(RNNCell):
  """Basic attention cell wrapper.

  Implementation based on https://arxiv.org/abs/1409.0473.
  """

  def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, reuse=None):
    """Create a cell with attention.

    Paramters
    ---------
    cell: an RNNCell, an attention is added to it.
    attn_length: integer, the size of an attention window.
    attn_size: integer, the size of an attention vector. Equal to
        cell.output_size by default.
    attn_vec_size: integer, the number of convolutional features calculated
        on attention state and a size of the hidden layer built from
        base cell state. Equal attn_size to by default.
    input_size: integer, the size of a hidden linear layer,
        built from inputs and attention. Derived from the input tensor
        by default.
    state_is_tuple: If True, accepted and returned states are n-tuples, where
      `n = len(cells)`.  By default (False), the states are all
      concatenated along the column axis.
    reuse: (optional) Python boolean describing whether to reuse variables
      in an existing scope.  If not `True`, and the existing scope already has
      the given variables, an error is raised.

    Return
    ------
    output, (new_state, new_attns, new_attn_states)

    Raises
    ------
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    super(AttentionCell, self).__init__(_reuse=reuse)
    if not isinstance(cell, RNNCell):  # pylint: disable=protected-access
      raise TypeError("The parameter cell is not RNNCell.")
    if attn_length <= 0:
      raise ValueError("attn_length should be greater than zero, got %s"
                       % str(attn_length))
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse

  @property
  def state_size(self):
    size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length)
    return size

  @property
  def output_size(self):
    return self._attn_size

  def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""
    state, attns, attn_states = state
    attn_states = array_ops.reshape(attn_states,
                                    [-1, self._attn_length, self._attn_size])
    input_size = self._input_size
    if input_size is None:
      input_size = inputs.shape.as_list()[1]
    inputs = _linear([inputs, attns], input_size, True)
    lstm_output, new_state = self._cell(inputs, state)
    new_state_cat = array_ops.concat(nest.flatten(new_state), 1)
    new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
    with tf.variable_scope("attn_output_projection"):
      output = _linear([lstm_output, new_attns], self._attn_size, True)
    new_attn_states = array_ops.concat(
        [new_attn_states, array_ops.expand_dims(output, 1)], 1)
    new_attn_states = array_ops.reshape(
        new_attn_states, [-1, self._attn_length * self._attn_size])
    new_state = (new_state, new_attns, new_attn_states)
    return output, new_state

  def _attention(self, query, attn_states):
    conv2d = nn_ops.conv2d
    reduce_sum = math_ops.reduce_sum
    softmax = nn_ops.softmax
    tanh = math_ops.tanh

    with tf.variable_scope("attention"):
      k = tf.get_variable(
          "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
      v = tf.get_variable("attn_v", [self._attn_vec_size])
      hidden = array_ops.reshape(attn_states,
                                 [-1, self._attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      y = _linear(query, self._attn_vec_size, True)
      y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = array_ops.reshape(d, [-1, self._attn_size])
      new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
      return new_attns, new_attn_states
