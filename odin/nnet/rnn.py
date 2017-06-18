from __future__ import division, absolute_import, print_function

from abc import ABCMeta, abstractmethod
from six import add_metaclass
from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

from odin import backend as K
from odin.backend.role import (InitialState, Weight, Bias, Parameter,
                        has_roles, BatchNormShiftParameter,
                        BatchNormScaleParameter,
                        BatchNormPopulationMean,
                        BatchNormPopulationInvStd)
from odin.utils import as_tuple

from .base import NNOp, _nnops_initscope
from .helper import Sequence, HelperOps
from .normalization import BatchNorm


# ===========================================================================
# Helper
# ===========================================================================
def _init_input2hidden(ops, rnn_mode, input_mode, W_init, input_dims, hidden_dims):
    # N represent the number of gates
    if 'rnn' in rnn_mode:
        N = 1
        msg = '(W_hid)'
    elif rnn_mode == 'gru':
        N = 3
        msg = '(W_input_to_updategate, W_input_to_resetgate, W_input_to_hiddenupdate)'
    elif rnn_mode == 'lstm':
        N = 4
        msg = '(W_input_to_inputgate, W_input_to_forgetgate, W_input_to_hidden, W_input_to_outputgate)'
    # ====== check input ====== #
    if input_mode != 'skip':
        ops.config.create_params(W_init, shape=(input_dims, hidden_dims),
                             name='W_in', roles=Weight, nb_params=N)
        if input_mode == 'norm':
            ops.config.create_params(K.rand.constant(0.), shape=(hidden_dims * N,),
                                 name='beta', roles=BatchNormShiftParameter)
            ops.config.create_params(K.rand.constant(1.), shape=(hidden_dims * N,),
                                 name='gamma', roles=BatchNormScaleParameter)
            ops.config.create_params(K.rand.constant(0.), shape=(hidden_dims * N,),
                                 name='mean', roles=BatchNormPopulationMean)
            ops.config.create_params(K.rand.constant(1.), shape=(hidden_dims * N,),
                                 name='inv_std', roles=BatchNormPopulationInvStd)
    # skip input mode
    elif input_dims != hidden_dims and \
    input_dims != hidden_dims * N: # 3 gates + 1 hid_update
        raise Exception('Skip input mode, input trailing_dimension=%d '
                        '(the final dim) must equal to the number of hidden '
                        'units (tied input connection), or %d-th the number '
                        'of hidden units = %d, which include: ' + msg %
                        (input_dims, N, hidden_dims * N))


def _check_cudnn_hidden_init(s0, shape, nnops, name):
    nb_layers, batch_size, hidden_size = shape
    # ====== init s0 ====== #
    if s0 is None and hasattr(nnops, name):
        s0 = getattr(nnops, name)
    elif s0 is not None:
        if callable(s0) or K.is_trainable_variable(s0) or isinstance(s0, np.ndarray):
            _ = (nb_layers, 1, hidden_size) if callable(s0) or isinstance(s0, np.ndarray) \
                else s0.get_shape()
            s0 = nnops.config.create_params(
                s0, shape=_, name=name, roles=InitialState)
        # ====== check s0 shape ====== #
        init_shape = s0.get_shape()
        if s0.get_shape().ndims == 2:
            if K.get_shape(s0)[-1] != hidden_size:
                raise ValueError('init state has %d dimension, but the hidden_size=%d' %
                                (init_shape[-1], hidden_size))
        elif init_shape[::2] != (nb_layers, hidden_size):
            raise ValueError('Require init states of size: %s, but '
                             'given state of size: %s' % (shape, init_shape))
        # ====== return the right shape ====== #
        setattr(nnops, name, s0)
    return s0


# ===========================================================================
# Dynamic RNN
# ===========================================================================
class DynamicRNN(NNOp):
    # cell_fw, cell_bw, inputs, sequence_length=None,
    # initial_state_fw=None, initial_state_bw=None,
    # dtype=None, parallel_iterations=None,
    # swap_memory=False, time_major=False, scope=None

    @_nnops_initscope
    def __init__(self, cell, state=None,
                 cell_bw=None, state_bw=None,
                 return_states=False, **kwargs):
        super(DynamicRNN, self).__init__(**kwargs)
        self.cell = cell
        self.cell_bw = cell_bw
        self.state = state
        self.state_bw = state_bw
        self.return_states = return_states

    def _initialize(self):
        pass

    def _apply(self, X, state=None, state_bw=None):
        # time_major: The shape format of the `inputs` and `outputs` Tensors.
        #   If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        #   If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        if self.cell_bw is None:
            outputs = rnn.dynamic_rnn(self.cell, inputs=X,
                initial_state=state if self.state is None else self.state,
                dtype=X.dtype.base_dtype,
                time_major=False)
            self.cell._reuse = True
        else:
            outputs = rnn.bidirectional_dynamic_rnn(
                cell_fw=self.cell, cell_bw=self.cell_bw, inputs=X,
                initial_state_fw=state if self.state is None else self.state,
                initial_state_bw=state_bw if self.state_bw is None else self.state_bw,
                time_major=False)
            self.cell._reuse = True
            self.cell_bw._reuse = True
        K.eval(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)))
        if not self.return_states:
            return outputs[0]
        return outputs

    def _transpose(self):
        raise NotImplementedError


# ===========================================================================
# Static RNN
# ===========================================================================
class StaticRNN(NNOp):
    # cell_fw, cell_bw, inputs, sequence_length=None,
    # initial_state_fw=None, initial_state_bw=None,
    # dtype=None, parallel_iterations=None,
    # swap_memory=False, time_major=False, scope=None

    @_nnops_initscope
    def __init__(self, cell_fw, cell_bw=None,
                 state_fw=None, state_bw=None, **kwargs):
        super(StaticRNN, self).__init__(**kwargs)
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        self.state_fw = state_fw
        self.state_bw = state_bw

    def _initialize(self):
        if self.state_fw is not None:
            pass
        if self.state_bw is not None:
            pass

    def _apply(self, X, states):
        if self.cell_bw is None:
            rnn.static_rnn
        else:
            rnn.static_bidirectional_rnn

    def _transpose(self):
        pass


# ===========================================================================
# DNN
# ===========================================================================
class CudnnRNN(NNOp):

    """CuDNN v5 RNN implementation.

    Parameters
    ----------
    num_units : int
        the number of units within the RNN model.
    W_init:
        initial description for weights
    b_init:
        initial description for bias
    rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
        See cudnn documentation for ``cudnnRNNMode_t``.
    num_layers : int
        the number of layers for the RNN model.
    input_mode : {'linear', 'skip', 'norm'}
        linear: input will be multiplied by a biased matrix.
        norm: same as linear, but batch norm will be added for input connection
        skip: No operation is performed on the input.  The size must
        match the hidden size. (CuDNN docs: cudnnRNNInputMode_t)
        norm: applying batch normalization on input-to-hidden connection, this
        approach require the `input_dims` equal to `num_units`.
    direction_mode : {'unidirectional', 'bidirectional'}
        unidirectional: The network operates recurrently from the
                        first input to the last.
        bidirectional: The network operates from first to last then from last
                       to first and concatenates the results at each layer.
    params_split: boolean (defaults: False)
        if True, separately initialized each parameter of RNN, then flatten and
        concatenate all of them into one big vector for Cudnn, this results
        more flexible control over parameters but significantly reduce the
        speed.
    return_states: boolean (defaults: False)
        if True, this Ops returns the [output, hidden_staes, cell_states (lstm)]
        otherwise only return the output
    dropout: float (0.0-1.0)
        whether to enable dropout. With it is 0, dropout is disabled.

    Returns
    -------
    [output, hidden_states, cell_states] for lstm
    [output, hidden_states] for gru and rnn

    output_shape: (batch_size, timesteps,  num_units)
    hidden_shape: (num_layers, batch_size, num_units)
    cell_shape: (num_layers, batch_size,   num_units)

    """

    @_nnops_initscope
    def __init__(self, num_units,
            W_init=K.rand.glorot_uniform,
            b_init=K.rand.constant(0.),
            rnn_mode='lstm', num_layers=1,
            input_mode='linear',
            direction_mode='unidirectional',
            params_split=False,
            return_states=False,
            dropout=0., **kwargs):
        super(CudnnRNN, self).__init__(**kwargs)
        # ====== defaults recurrent control ====== #
        self.num_units = int(num_units)
        self.num_layers = int(num_layers)
        self.rnn_mode = rnn_mode
        self.input_mode = input_mode
        self.direction_mode = direction_mode
        self.params_split = params_split
        self.return_states = return_states
        self.dropout = dropout

        if not callable(W_init):
            raise ValueError('W_init must be callable with input is variable shape')
        self.W_init = W_init
        if not callable(b_init):
            raise ValueError('b_init must be callable with input is variable shape')
        self.b_init = b_init

    # ==================== abstract methods ==================== #
    def _transpose(self):
        # flip the input and hidden
        raise NotImplementedError

    def _initialize(self):
        input_shape = self.input_shape
        is_bidirectional = self.direction_mode == 'bidirectional'
        # ====== check input ====== #
        if self.input_mode == 'norm':
            _init_input2hidden(self, rnn_mode=self.rnn_mode, input_mode=self.input_mode,
                               W_init=self.W_init, input_dims=input_shape[-1],
                               hidden_dims=self.num_units)
        # ====== create params ====== #
        layer_info = [input_shape[-1], self.num_units] + \
                     [self.num_units * (2 if is_bidirectional else 1),
                      self.num_units] * (self.num_layers - 1)
        if self.rnn_mode == 'lstm':
            from odin.backend.init import lstm as init_func
        elif self.rnn_mode == 'gru':
            from odin.backend.init import gru as init_func
        else:
            from odin.backend.init import rnn as init_func
        # initialize each parameter in params_split=True
        if self.params_split:
            with tf.variable_scope(self.name):
                parameters = [init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                        W_init=self.W_init, b_init=self.b_init,
                                        one_vector=False, return_variable=True,
                                        bidirectional=is_bidirectional,
                                        name='layer%d' % i)
                              for i in range(self.num_layers)]
            # print([(j.name, j.tag.roles) for i in parameters for j in i]); exit()
            for p in chain(*parameters):
                self.config.create_params(p, shape=p.get_shape(),
                                     name=p.name.split(':')[0].split('/')[1],
                                     roles=Parameter)
        # else initialize all in 1 big vector
        else:
            parameters = np.concatenate([init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                         one_vector=True, return_variable=False,
                                         bidirectional=is_bidirectional)
                                         for i in range(self.num_layers)])
            self.config.create_params(parameters, shape=parameters.shape,
                                      name='params', roles=Parameter)

    def _apply(self, X, h0=None, c0=None, mask=None):
        batch_size = X.get_shape()[0]
        is_bidirectional = self.direction_mode == 'bidirectional'
        input_mode = ('skip' if self.input_mode == 'skip' or self.input_mode == 'norm'
                      else 'linear')
        # ====== precompute input ====== #
        # linear or norm input mode
        if self.input_mode == 'norm':
            X = K.dot(X, self.W_in)
            # normalize all axes except the time dimension
            bn = BatchNorm(axes=(0, 1), activation=K.linear,
                           gamma_init=self.gamma, beta_init=self.beta,
                           mean_init=self.mean, inv_std_init=self.inv_std)
            X = bn(X)
            # cudnnRNN doesnt' support multiple inputs
            shapeX = X.get_shape()
            ndims = shapeX.ndims
            if 'rnn' in self.rnn_mode: N = 1
            elif self.rnn_mode == 'gru': N = 3
            else: N = 4
            newshape = [shapeX[i] for i in range(ndims - 1)] + [self.num_units, N]
            X = tf.reduce_mean(K.reshape(X, newshape), axis=-1)
        # ====== hidden state ====== #
        num_layers = self.num_layers * 2 if is_bidirectional else self.num_layers
        require_shape = (num_layers, batch_size, self.num_units)
        h0 = _check_cudnn_hidden_init(h0, require_shape, self, 'h0')
        c0 = _check_cudnn_hidden_init(c0, require_shape, self, 'c0')
        # ====== parameters ====== #
        if self.params_split:
            parameters = tf.concat([K.flatten(i, outdim=1)
                                    for i in self.parameters
                                    if not has_roles(i, InitialState)])
        else:
            parameters = self.params
        # ====== return CuDNN RNN ====== #
        results = K.rnn_dnn(X, hidden_size=self.num_units, rnn_mode=self.rnn_mode,
                           num_layers=self.num_layers, parameters=parameters,
                           h0=h0, c0=c0, input_mode=input_mode,
                           direction_mode=self.direction_mode,
                           dropout=self.dropout, name=self.name)
        if not self.return_states:
            results = results[0] # only get the output
        return results
