from __future__ import print_function, division, absolute_import

import numpy as np

from odin import backend as K, utils

from .base import Dense
from .rnn import LSTM
from .normalization import BatchNorm
from .helper import Sequence, Merge


def lstm_batch_norm(num_units,
                    W_input_init=K.init.glorot_uniform,
                    W_hidden_init=K.init.orthogonal,
                    W_peephole_init=K.init.glorot_uniform,
                    activation=K.tanh, gate_activation=K.sigmoid,
                    tied_input=False, batch_norm=True, name=None):
    if name is None:
        name = 'lstm_batch_norm_%s' % utils.uuid()
    # ====== create input_gates ====== #
    ops_list = []
    bias = None if batch_norm else K.init.constant(0)
    if tied_input:
        input_gates = Dense(num_units, W_init=W_input_init, b_init=bias,
                            activation=K.linear, name='%s_gates' % name)
    else:
        input_gates = Merge([
            Dense(num_units, W_init=W_input_init, b_init=bias, activation=K.linear,
                  name='%s_ingate' % name), # input-gate
            Dense(num_units, W_init=W_input_init, b_init=bias, activation=K.linear,
                  name='%s_forgetgate' % name), # forget-gate
            Dense(num_units, W_init=W_input_init, b_init=bias, activation=K.linear,
                  name='%s_cellupdate' % name), # cell-update
            Dense(num_units, W_init=W_input_init, b_init=bias, activation=K.linear,
                  name='%s_outgate' % name) # output-gate
        ], merge_function=K.concatenate)
    ops_list.append(input_gates)
    # ====== batch_norm ====== #
    # normalize batch and time dimension
    if batch_norm:
        ops_list.append(BatchNorm(axes=(0, 1), name='%s_norm' % name))
    # ====== add LSTM ====== #
    ops_list.append(LSTM(num_units=num_units,
                         activation=activation,
                         gate_activation=gate_activation,
                         W_init=W_hidden_init,
                         W_peepholes=W_peephole_init,
                         name='%s_lstm' % name))
    return Sequence(ops_list, name=name)
