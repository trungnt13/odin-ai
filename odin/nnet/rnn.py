from __future__ import division, absolute_import, print_function

import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
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
from odin.utils import as_tuple, is_string, is_number

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
        if callable(s0) or K.is_variable(s0) or isinstance(s0, np.ndarray):
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
def get_cell_info(cell):
    from tensorflow.contrib.rnn.python.ops import rnn_cell
    found_cell = None
    for key, val in inspect.getmembers(K.rnn_cell) + inspect.getmembers(rnn_cell):
        if inspect.isclass(val) and issubclass(val, tf.contrib.rnn.RNNCell):
            if (isinstance(cell, str) and (key == cell or str(val) == cell)) \
            or cell == val:
                found_cell = val; break
    # ====== get cell init info ====== #
    _ = inspect.getargspec(found_cell.__init__)
    args = _.args[1:]
    kwargs = {}
    if _.defaults is not None:
        kwargs = {i: j for i, j in zip(args[::-1], _.defaults[::-1])}
    return found_cell, args, kwargs


class RNN(NNOp):
    """
    Parameter
    ---------
    attention: None or `tf.contrib.seq2seq.AttentionMechanism`
        two basic attentions: LuongAttention, BahdanauAttention
    """

    @_nnops_initscope
    def __init__(self, cell_type, kwargs={}, attention=None,
                 num_layers=1, dynamic=True, return_states=False,
                 name=None):
        super(RNN, self).__init__(name=name)
        self.cell_type, _args, _kwargs = get_cell_info(cell_type)
        if self.cell_type is None:
            raise RuntimeError("Cannot find any RNNCell with given description: %s"
                % str(cell_type))
        _kwargs.update(kwargs)
        if len(_kwargs) != len(_args):
            raise RuntimeError("Missing following arguments: %s for the RNNCell "
                "of type: %s" % (list(set(_args) - set(_kwargs.keys())),
                                 self.cell_type.__name__))
        self.cell_kwargs = _kwargs
        self._num_layers = int(num_layers)
        self.dynamic = dynamic
        self.return_states = return_states
        self._initialize_variables = False
        # ====== attention ====== #
        if attention is not None and \
        not issubclass(attention, tf.contrib.seq2seq.AttentionMechanism):
            raise ValueError("`attention` argument must be `None` or instance "
                "of `tensorflow.contrib.seq2seq.AttentionMechanism`.")
        self.attention = attention
        tf.contrib.seq2seq.embedding_attention_seq2seq

    @property
    def cell(self):
        # ====== first time create Cell ====== #
        if not hasattr(self, '_cell'):
            c = self.cell_type(**self.cell_kwargs)
            if self._num_layers > 1:
                c = tf.contrib.rnn.MultiRNNCell([c] * self._num_layers)
            self._cell = c
        # ====== return Cell ====== #
        self._cell._reuse = False if len(self.variables) == 0 else True
        return self._cell

    def _apply(self, X, state=None, memory=None):
        # time_major: The shape format of the `inputs` and `outputs` Tensors.
        #   If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        #   If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        rnn_func = rnn.dynamic_rnn if self.dynamic else rnn.static_rnn
        cell = self.cell
        # ====== create attention if necessary ====== #
        if self.attention is not None:
            if not hasattr(self, "cell_with_attention"):
                # normalize=True, probability_fn=tf.nn.softmax
                attention_mechanism = self.attention(num_units=128, memory=memory)
                self.cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism,
                    attention_layer_size=attention_mechanism._num_units)
            cell = self.cell_with_attention
        outputs = rnn_func(cell, inputs=X, initial_state=state,
                           dtype=X.dtype.base_dtype)
        # ====== initialize cell ====== #
        if not self._initialize_variables:
            # initialize only once, everytime you call this, the values of
            # variables changed
            K.eval(tf.variables_initializer(self.variables))
            self._initialize_variables = True
        # ====== return ====== #
        if not self.return_states:
            return outputs[0]
        return outputs

    def _transpose(self):
        raise NotImplementedError


class LSTM(RNN):

    @_nnops_initscope
    def __init__(self, num_units, use_peepholes=False, cell_clip=None,
                 num_proj=None, proj_clip=None, forget_bias=1.0,
                 activation=None, num_layers=1, attention=None,
                 dynamic=True, return_states=False, name=None):
        super(LSTM, self).__init__(cell_type=tf.contrib.rnn.LSTMCell,
            kwargs={'num_units': num_units, 'use_peepholes': use_peepholes,
                    'cell_clip': cell_clip, 'num_proj': num_proj,
                    'proj_clip': proj_clip, 'forget_bias': forget_bias,
                    'activation': activation},
            num_layers=num_layers, attention=attention,
            dynamic=dynamic, return_states=return_states,
            name=name)


class GRU(RNN):

    @_nnops_initscope
    def __init__(self, num_units, activation=None, attention=None,
                 num_layers=1, dynamic=True, return_states=False, name=None):
        super(GRU, self).__init__(cell_type=tf.contrib.rnn.GRUCell,
            kwargs={'num_units': num_units, 'activation': activation},
            num_layers=num_layers, attention=attention,
            dynamic=dynamic, return_states=return_states,
            name=name)


class BiRNN(RNN):

    @_nnops_initscope
    def __init__(self, forward, backward, dynamic=True,
                 return_states=False, name=None):
        super(BiRNN, self).__init__(dynamic=dynamic, return_states=return_states,
                                    name=name)
        if not isinstance(forward, RNN):
            raise ValueError("`forward` must be instance of `odin.rnn.RNN`")
        self.forward = forward

        if not isinstance(backward, RNN):
            raise ValueError("`backward` must be instance of `odin.rnn.RNN`")
        self.backward = backward

    @property
    def cell(self):
        return [self.forward.cell, self.backward.cell]

    def _apply(self, X, state=None, state_bw=None):
        # time_major: The shape format of the `inputs` and `outputs` Tensors.
        #   If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        #   If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        rnn_func = rnn.bidirectional_dynamic_rnn if self.dynamic \
            else rnn.static_bidirectional_rnn
        outputs = rnn_func(inputs=X,
            cell_fw=self.forward.cell, cell_bw=self.backward.cell,
            initial_state_fw=state, initial_state_bw=state_bw,
            dtype=X.dtype.base_dtype)
        self.forward.cell._reuse = True
        self.backward.cell._reuse = True
        if not self.return_states:
            return outputs[0]
        return outputs


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
            from odin.backend.rand import lstm as init_func
        elif self.rnn_mode == 'gru':
            from odin.backend.rand import gru as init_func
        else:
            from odin.backend.rand import rnn as init_func
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
                self.get_variable(initializer=p, shape=p.get_shape(),
                                  name=p.name.split(':')[0].split('/')[1],
                                  roles=Parameter)
        # else initialize all in 1 big vector
        else:
            parameters = np.concatenate([init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                         one_vector=True, return_variable=False,
                                         bidirectional=is_bidirectional)
                                         for i in range(self.num_layers)])
            self.get_variable(initializer=parameters, shape=parameters.shape,
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
