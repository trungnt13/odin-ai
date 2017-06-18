from __future__ import absolute_import, division, print_function

import collections
import math

import tensorflow as tf

from tensorflow.contrib.rnn import (RNNCell, MultiRNNCell,
    DropoutWrapper, EmbeddingWrapper,
    InputProjectionWrapper, OutputProjectionWrapper,
    PhasedLSTMCell, GridLSTMCell, GLSTMCell,
    IntersectionRNNCell, UGRNNCell, NASCell,
    LayerNormBasicLSTMCell, HighwayWrapper,
    AttentionCellWrapper, BidirectionalGridLSTMCell,
    TimeFreqLSTMCell, CoupledInputForgetGateLSTMCell)
from tensorflow.python.ops.rnn_cell_impl import (_linear,
    GRUCell, LSTMCell, BasicLSTMCell)
from tensorflow.python.util import nest


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
        output = self._activation(
            _linear([inputs, state], output_size=self._num_units,
            bias=False if self._bias_initializer is None else True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer))
        return output, output
