# This code make sure the GPU implementation of GRU and LSTM using CuDNN
# is used in tensorflow 2.0, also ensure a similer interface to RNN as in
# `odin.network_torch`
#
# For more information:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.keras import Model, layers


class _RNNModel(Model):

  def call(self, inputs, **kwargs):
    outputs = self._rnn(inputs)
    if isinstance(outputs, (tuple, list)):
      outputs = [outputs[0]] + [tf.expand_dims(o, axis=0) for o in outputs[1:]]
      if isinstance(self._rnn, layers.Bidirectional):
        outputs = [outputs[0]] + [
            tf.concat([outputs[i], outputs[i + 1]], axis=0)
            for i in range(1, len(outputs), 2)
        ]
    return outputs


class SimpleRNN(_RNNModel):

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
    super(SimpleRNN, self).__init__(*kwargs)
    assert num_layers == 1, "Only support single layer for CuDNN RNN in keras"
    self._rnn = layers.SimpleRNN(units=units,
                                 activation=activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 recurrent_initializer=recurrent_initializer,
                                 bias_initializer=bias_initializer,
                                 dropout=dropout,
                                 recurrent_dropout=0.,
                                 return_sequences=return_sequences,
                                 return_state=return_state,
                                 go_backwards=go_backwards,
                                 stateful=stateful,
                                 unroll=False)
    if bidirectional:
      self._rnn = layers.Bidirectional(
          self._rnn,
          merge_mode='concat',
      )


class GRU(_RNNModel):

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
    super(GRU, self).__init__(**kwargs)
    assert num_layers == 1, "Only support single layer for CuDNN RNN in keras"
    self._rnn = layers.GRU(
        # cuDNN requirement
        activation='tanh',
        recurrent_activation='sigmoid',
        recurrent_dropout=0,
        unroll=False,
        use_bias=use_bias,
        reset_after=True,
        # free arguments
        units=units,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        dropout=dropout,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)
    if bidirectional:
      self._rnn = layers.Bidirectional(
          self._rnn,
          merge_mode='concat',
      )


class LSTM(_RNNModel):

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
    super(LSTM, self).__init__(**kwargs)
    assert num_layers == 1, "Only support single layer for CuDNN RNN in keras"
    self._rnn = layers.LSTM(
        # cuDNN requirement
        activation='tanh',
        recurrent_activation='sigmoid',
        recurrent_dropout=0,
        unroll=False,
        use_bias=use_bias,
        # free arguments
        units=units,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        dropout=dropout,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)
    if bidirectional:
      self._rnn = layers.Bidirectional(
          self._rnn,
          merge_mode='concat',
      )
