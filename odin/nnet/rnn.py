from __future__ import division, absolute_import, print_function

import inspect
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from itertools import chain

import numpy as np

from odin import backend as K
from odin.basic import (INITIAL_STATE, WEIGHT, BIAS, PARAMETER,
                        has_roles, BATCH_NORM_SHIFT_PARAMETER,
                        BATCH_NORM_SCALE_PARAMETER,
                        BATCH_NORM_POPULATION_MEAN,
                        BATCH_NORM_POPULATION_INVSTD)
from odin.utils import as_tuple

from .base import NNConfig, NNOps
from .helper import Sequence, HelperOps
from .normalization import BatchNorm


# ===========================================================================
# Helper
# ===========================================================================
def _slice_x(X, idx):
    """ Slice tensor at its last dimension """
    ndim = K.ndim(X)
    _ = [slice(None, None, None) for i in range(ndim - 1)]
    return X[_ + [idx]]


def _check_rnn_hidden_states(h0, ops, input_shape, name):
    if h0 is None and hasattr(ops, name):
        h0 = getattr(ops, name)
    else:
        h0 = K.init.constant(0.) if h0 is None else h0
        # only store trainable variable or constant
        if callable(h0) or K.is_trainable_variable(h0) or isinstance(h0, np.ndarray):
            h0 = ops.configuration.create_params(h0,
                shape=(1,) + input_shape[2:-1] + (ops.num_units,),
                name=name, nnops=ops, roles=INITIAL_STATE)
        else: # still store the states so it can be re-used on other inputs
            ops.h0 = h0
    return h0


def _init_input2hidden(ops, config, rnn_mode, input_mode,
                       W_init, input_dims, hidden_dims):
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
        config.create_params(W_init, shape=(input_dims, hidden_dims),
                             name='W_in', nnops=ops, roles=WEIGHT,
                             nb_params=N)
        if input_mode == 'norm':
            config.create_params(K.init.constant(0.), shape=(hidden_dims * N,),
                                 name='beta', nnops=ops,
                                 roles=BATCH_NORM_SHIFT_PARAMETER)
            config.create_params(K.init.constant(1.), shape=(hidden_dims * N,),
                                 name='gamma', nnops=ops,
                                 roles=BATCH_NORM_SCALE_PARAMETER)
            config.create_params(K.init.constant(0.), shape=(hidden_dims * N,),
                                 name='mean', nnops=ops,
                                 roles=BATCH_NORM_POPULATION_MEAN)
            config.create_params(K.init.constant(1.), shape=(hidden_dims * N,),
                                 name='inv_std', nnops=ops,
                                 roles=BATCH_NORM_POPULATION_INVSTD)
    # skip input mode
    elif input_dims != hidden_dims and \
    input_dims != hidden_dims * N: # 3 gates + 1 hid_update
        raise Exception('Skip input mode, input trailing_dimension=%d '
                        '(the final dim) must equal to the number of hidden '
                        'units (tied input connection), or %d-th the number '
                        'of hidden units = %d, which include: ' + msg %
                        (input_dims, N, hidden_dims * N))


@add_metaclass(ABCMeta)
class BaseRNN(NNOps):

    def __init__(self, **kwargs):
        super(BaseRNN, self).__init__(**kwargs)
        # ====== defaults recurrent control ====== #
        self.repeat_states = kwargs.pop('repeat_states', True)
        self.iterate = kwargs.pop('iterate', True)
        self.backwards = kwargs.pop('backwards', False)
        self.n_steps = kwargs.pop('n_steps', None)
        self.batch_size = kwargs.pop('batch_size', None)

    @abstractmethod
    def _rnn(self, **kwargs):
        pass

    def get_recurrent_info(self, kwargs):
        """ Return information that control how this ops recurrently
        performed

        Parameters
        ----------
        kwargs: keywords arguments
            all arguments given that will override default configuration

        """
        kwargs = kwargs if isinstance(kwargs, dict) else {}
        return {
            'iterate': kwargs.pop('iterate', self.iterate),
            'backwards': kwargs.pop('backwards', self.backwards),
            'repeat_states': kwargs.pop('repeat_states', self.repeat_states),
            'name': kwargs.pop('name', self.name),
            'n_steps': kwargs.pop('n_steps', self.n_steps),
            'batch_size': kwargs.pop('batch_size', self.batch_size),
        }


class BidirectionalRNN(HelperOps):
    """ BidirectionalRNN

    Parameters
    ----------
    forward: NNOps
        forward network
    backward: NNOps
        the backward network
    mode: callable
        a function to merge the output of forward and backward networks
    """

    def __init__(self, forward, backward=None, mode=K.concatenate, **kwargs):
        if not isinstance(forward, NNOps):
            raise ValueError('forward must be instance of NNOps, but it is %s'
                            % str(forward.__class__))
        if backward is None:
            if isinstance(forward, RNN):
                backward = RNN(num_units=forward.num_units,
                               activation=forward.activation,
                               W_init=forward.W_init,
                               b_init=forward.b_init,
                               input_mode=forward.input_mode,
                               name=forward.name + '_backward',
                               backwards=not forward.backwards)
            elif isinstance(forward, GRU):
                backward = GRU(num_units=forward.num_units,
                               activation=forward.activation,
                               gate_activation=forward.gate_activation,
                               W_in_init=forward.W_in_init,
                               W_hid_init=forward.W_hid_init,
                               b_init=forward.b_init,
                               input_mode=forward.input_mode,
                               name=forward.name + '_backward',
                               backwards=not forward.backwards)
            elif isinstance(forward, LSTM):
                backward = LSTM(num_units=forward.num_units,
                                activation=forward.activation,
                                gate_activation=forward.gate_activation,
                                W_in_init=forward.W_in_init,
                                W_hid_init=forward.W_hid_init,
                                W_peepholes=forward.W_peepholes,
                                b_init=forward.b_init,
                                input_mode=forward.input_mode,
                                return_cell_memory=forward.return_cell_memory,
                                name=forward.name + '_backward',
                                backwards=not forward.backwards)
            elif isinstance(forward, CudnnRNN):
                raise NotImplementedError
            else:
                raise Exception('No support for auto-infer backward of %s' %
                                str(forward.__class__))
        super(BidirectionalRNN, self).__init__(ops=[forward, backward], **kwargs)
        # ====== check mode ====== #
        if isinstance(mode, str):
            if 'concat' in mode.lower():
                mode = K.concatenate
            elif any(i in mode.lower() for i in ['sum', 'add']):
                mode = K.add
        if not callable(mode):
            raise ValueError("mode must be callable with two input arguments, "
                             "which are output of forward and backward ops.")
        self.mode = mode

    def _initialize(self, X):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape)
        return config

    def _apply(self, X, h0=None, c0=None, mask=None, **kwargs):
        forward = self.ops[0].apply(X, h0=h0, c0=c0, mask=mask, **kwargs)
        backward = self.ops[1].apply(X, h0=h0, c0=c0, mask=mask, **kwargs)
        return_list = False
        if isinstance(forward, (tuple, list)) or isinstance(backward, (tuple, list)):
            return_list = True
        results = list(zip(as_tuple(forward), as_tuple(backward)))
        # post processing the outputs:
        results = [self.mode(r[0], r[1]) if self.mode in (K.add, K.sub, K.mul, K.div, K.mod)
                   else self.mode(r)
                   for r in results]
        if not return_list:
            results = results[0]
        return results


# ===========================================================================
# RNN
# ===========================================================================
class RNN(BaseRNN):

    """
    Parameters
    ----------
    num_units: int
        number of hidden units
    activation: callable
        activate function for output of RNN
    W_init : variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 2D matrix with shape
    b_init : variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``
    input_mode : {'linear', 'skip', 'norm'}
        linear: input will be multiplied by a biased matrix
        norm: same as linear, but batch norm will be added for input connection
        skip: No operation is performed on the input.  The size must
        match the hidden size.
        (CuDNN docs: cudnnRNNInputMode_t)

    Example
    -------
    >>> import numpy as np
    >>> from odin import backend as K, nnet as N
    >>> def random(*shape):
    ...     return np.random.rand(*shape).astype(CONFIG['floatX']) / 12
    >>> def random_bin(*shape):
    ...     return np.random.randint(0, 2, size=shape).astype('int32')
    >>> W = [random(28, 32), random(32, 32), random(32), random_bin(12, 28)]
    >>> f = N.Sequence([
    ...     N.Dense(num_units=32, W_init=W[0], b_init=W[2],
    ...         activation=K.linear),
    ...     N.RNN(num_units=32, activation=K.relu,
    ...         W_init=W[1])
    >>> ])
    >>> return X1, f(X1, hid_init=zeros(1, 32))[0]

    """

    def __init__(self, num_units, activation=K.relu,
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0.),
                 input_mode='linear', **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = K.linear if activation is None else activation
        self.W_init = W_init
        self.b_init = b_init
        self.input_mode = input_mode

    @K.rnn_decorator(sequences=['X', 'mask'], states=['h0'])
    def _rnn(self, X, h0, mask=None):
        bias = 0. if self.b_init is None else self.b
        next_states = self.activation(X + K.dot(h0, self.W_hid) + bias)
        if mask is not None:
            next_states = K.switch(mask, next_states, h0)
        return next_states

    def _apply(self, X, h0=None, mask=None, **kwargs):
        input_shape = K.get_shape(X)
        # ====== check mask ====== #
        if mask is not None and (K.ndim(mask) != K.ndim(X) - 1 or
                                 K.get_shape(mask)[-1] != input_shape[1]):
            raise Exception('Mask must has "%d" dimensions and the time dimension '
                            '(i.e. the second dimension) must equal to "%d"'
                            ', but the given mask has shape "%s".' %
                            (K.ndim(X) - 1, input_shape[1], K.get_shape(mask)))
        # ====== initialize states ====== #
        h0 = _check_rnn_hidden_states(h0, self, input_shape, 'h0')
        # turn off repeat_states if batch_size already included
        if K.get_shape(h0)[0] != 1:
            self.repeat_states = False
        # ====== precompute input ====== #
        X = K.dot(X, self.W_in) if self.input_mode != 'skip' else X
        if self.input_mode == 'norm':
            # normalize all axes except the time dimension
            bn = BatchNorm(axes=(0, 1), activation=K.linear,
                           gamma_init=self.gamma, beta_init=self.beta,
                           mean_init=self.mean, inv_std_init=self.inv_std)
            X = bn(X)
        out = self._rnn(X, h0=h0, mask=mask,
                        **self.get_recurrent_info(kwargs))
        for i in out:
            K.add_shape(i, shape=tuple(input_shape[:-1]) + (self.num_units,))
        # only care about the first state
        return out[0] if len(out) == 1 else out

    def _initialize(self, X):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== initialize inner parameters ====== #
        W_init = as_tuple(self.W_init, N=2)
        _init_input2hidden(self, config, rnn_mode='rnn',
                           input_mode=self.input_mode,
                           W_init=W_init[0],
                           input_dims=input_shape[-1],
                           hidden_dims=self.num_units)
        # hidden connection
        config.create_params(W_init[1],
                             shape=(self.num_units, self.num_units),
                             name='W_hid',
                             nnops=self,
                             roles=WEIGHT)
        # bias
        if self.b_init is not None:
            config.create_params(self.b_init,
                                 shape=(self.num_units,),
                                 name='b',
                                 nnops=self,
                                 roles=BIAS)
        return config


# ===========================================================================
# GRU
# ===========================================================================
class GRU(BaseRNN):

    """
    Parameters
    ----------
    num_units: int
        pass
    activation: callable
        activation for hidden state
    gate_activation: callable
        activation for each gate
    W_in_init: variable, ndarray or callable (shape: (input_dims, num_units))
        Initializer for input-to-hidden weight matrices, if a list is given,
        the weights will be initialized in following order:
        "[W_input_to_updategate, W_input_to_resetgate, W_input_to_hiddenupdate]"
    W_hid_init: variable, ndarray or callable (shape: (num_units, num_units))
        Initializer for hidden-to-hidden weight matrices, if a list is given,
        the weights will be initialized in following order:
        "[W_hid_to_updategate, W_hid_to_resetgate, W_hid_to_hiddenupdate]"
    b_init : variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``
    input_mode : {'linear', 'skip', 'norm'}
        linear: input will be multiplied by a biased matrix
        norm: same as linear, but batch norm will be added for input connection
        skip: No operation is performed on the input.  The size must
        match the hidden size.
        (CuDNN docs: cudnnRNNInputMode_t)

    Example
    -------
    >>> import numpy as np
    >>> from odin import backend as K, nnet as N
    >>> def random(*shape):
    ...     return np.random.rand(*shape).astype(CONFIG['floatX']) / 12
    >>> W_in_to_updategate = random(28, 32)
    >>> W_hid_to_updategate = random(32, 32)
    >>> b_updategate = random(32)
    >>> W_in_to_resetgate = random(28, 32)
    >>> W_hid_to_resetgate = random(32, 32)
    >>> b_resetgate = random(32)
    >>> W_in_to_hidden_update = random(28, 32)
    >>> W_hid_to_hidden_update = random(32, 32)
    >>> b_hidden_update = random(32)
    >>> hid_init = random(1, 32)
    >>> x = random(12, 28, 28)
    >>> x_mask = np.random.randint(0, 2, size=(12, 28))
    >>> # create the network
    >>> X = K.placeholder(shape=(None, 28, 28), name='X')
    >>> mask = K.placeholder(shape=(None, 28), name='mask', dtype='int32')
    >>> f = N.Sequence([
    >>>     N.Merge([N.Dense(32, W_init=W_in_to_updategate, b_init=b_updategate,
    ...                      activation=K.linear, name='update'),
    >>>              N.Dense(32, W_init=W_in_to_resetgate, b_init=b_resetgate,
    ...                      activation=K.linear, name='reset'),
    >>>              N.Dense(32, W_init=W_in_to_hidden_update, b_init=b_hidden_update,
    ...                      activation=K.linear, name='hidden')],
    >>>             merge_function=K.concatenate),
    >>>     N.GRU(32, activation=K.tanh, gate_activation=K.sigmoid,
    >>>           W_init=[W_hid_to_updategate, W_hid_to_resetgate, W_hid_to_hidden_update],
    >>>           state_init=hid_init)
    >>> ])
    >>> y = f(X, mask=mask)
    >>> f = K.function([X, mask], y)
    >>> out1 = f(x, x_mask)[0]

    """

    def __init__(self, num_units,
                 activation=K.tanh,
                 gate_activation=K.sigmoid,
                 W_in_init=K.init.glorot_uniform,
                 W_hid_init=K.init.orthogonal,
                 b_init=K.init.constant(0.),
                 input_mode='linear',
                 **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = (K.tanh if activation is None
                           else activation)
        self.gate_activation = (K.sigmoid if gate_activation is None
                                else gate_activation)
        self.W_in_init = W_in_init
        self.W_hid_init = W_hid_init
        self.b_init = b_init
        self.input_mode = input_mode

    @K.rnn_decorator(sequences=['X', 'mask'], states=['h0'])
    def _rnn(self, X, h0, mask=None):
        #####################################
        # X: sequences inputs (included bias)
        # init: prev_states
        # W: concatenated [W_update, W_reset]
        # mask: mask inputs (optional)
        prev_states = h0
        nb_units = self.num_units
        # hidden connection of all gates and states update
        hid_connection = K.dot(prev_states, self.W_hid)
        # hidden to hidden connection
        hid_gate = _slice_x(hid_connection, slice(None, nb_units * 2))
        X_gate = _slice_x(X, slice(None, nb_units * 2))
        b_gate = 0 if self.b_init is None else _slice_x(self.b, slice(None, nb_units * 2))
        # states
        hid_states = _slice_x(hid_connection, slice(nb_units * 2, None))
        X_states = _slice_x(X, slice(nb_units * 2, None))
        b_states = 0 if self.b_init is None else _slice_x(self.b, slice(nb_units * 2, None))
        # new gates
        _ = self.gate_activation(X_gate + hid_gate + b_gate)
        update_values = _slice_x(_, slice(None, nb_units))
        reset_values = _slice_x(_, slice(nb_units, nb_units * 2))
        # calculate new gates
        new_states = self.activation(X_states + reset_values * hid_states + b_states)
        # final new states
        next_states = (new_states * update_values +
                       prev_states * (1 - update_values))
        # mask the next state
        if mask is not None:
            next_states = K.switch(mask, next_states, prev_states)
        return next_states

    def _apply(self, X, h0=None, mask=None, **kwargs):
        input_shape = K.get_shape(X)
        # ====== check mask ====== #
        if mask is not None and (K.ndim(mask) != 2 or
                                 K.get_shape(mask)[-1] != input_shape[1]):
            raise Exception('Mask must be a 2-D matrix and the time dimension '
                            '(i.e. the second dimension) must equal to "%d"'
                            ', but the given mask has shape "%s".' %
                            (input_shape[1], K.get_shape(mask)))
        # add broadcastable dimension for mask
        if mask is not None:
            mask = K.expand_dims(mask, dim=-1)
        # ====== initialize states ====== #
        h0 = _check_rnn_hidden_states(h0, self, input_shape, 'h0')
        # turn off repeat_states if batch_size already included
        if K.get_shape(h0)[0] != 1:
            self.repeat_states = False
        # ====== precompute inputs ====== #
        # linear or norm input mode
        if self.input_mode != 'skip':
            X = K.dot(X, self.W_in)
            if self.input_mode == 'norm':
                # normalize all axes except the time dimension
                bn = BatchNorm(axes=(0, 1), activation=K.linear,
                               gamma_init=self.gamma, beta_init=self.beta,
                               mean_init=self.mean, inv_std_init=self.inv_std)
                X = bn(X)
        # skip input
        elif input_shape[-1] == self.num_units:
            X = K.repeat(X, 3, axes=-1)
        # ====== compute recurrent output ====== #
        # recurrent
        out = self._rnn(X, h0=h0, mask=mask,
                        **self.get_recurrent_info(kwargs))
        for i in out:
            K.add_shape(i, shape=input_shape)
        # only care about the first state
        return out[0] if len(out) == 1 else out

    def _initialize(self, X):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== check input ====== #
        _init_input2hidden(self, config, rnn_mode='gru',
                           input_mode=self.input_mode,
                           W_init=self.W_in_init,
                           input_dims=input_shape[-1],
                           hidden_dims=self.num_units)
        # ====== initialize inner parameters ====== #
        # W_update, W_reset, W_hidden
        config.create_params(self.W_hid_init,
                             shape=(self.num_units, self.num_units),
                             name='W_hid',
                             nnops=self,
                             roles=WEIGHT,
                             nb_params=3)
        # bias
        if self.b_init is not None:
            config.create_params(self.b_init,
                                 shape=(self.num_units,),
                                 name='b',
                                 nnops=self,
                                 roles=BIAS,
                                 nb_params=3)
        return config


# ===========================================================================
# LSTM
# ===========================================================================
class LSTM(BaseRNN):

    """
            raise Exception('Skip input mode, input trailing_dimension=%d '
                            '(the final dim) must equal to the number of hidden '
                            'units (tied input connection), or 4-th the number '
                            'of hidden units (1 for W_input, 1 for W_forget, '
                            '1 for W_hidden, and 1 for W_output), which is: %d' %
                            (input_shape[-1], self.num_units * 4))

    Parameters
    ----------
    num_units: int
        pass
    activation: callable
        activation for hidden state
    gate_activation: callable
        activation for each gate
    W_in_init: variable, ndarray or callable (shape: (input_dims, num_units))
        Initializer for input-to-hidden weight matrices, if a list is given,
        the weights will be initialized in following order:
        "[W_input_to_inputgate, W_input_to_forgetgate, W_input_to_hidden, W_input_to_outputgate]"
    W_hid_init: variable, ndarray or callable (shape: (num_units, num_units))
        Initializer for hidden-to-hidden weight matrices, if a list is given,
        the weights will be initialized in following order:
        "[W_hid_to_inputgate, W_hid_to_forgetgate, W_hid_to_hidden, W_hid_to_outputgate]"
    W_peepholes: variable, ndarray or callable (shape: (num_units,))
        if `W_peepholes=None`, no peepholes are introduced. If a list is given,
        the weights will be initialized in following order:
        "[W_cell_to_inputgate, W_cell_to_forgetgate, W_cell_to_outputgate]""
    b_init : variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``
    input_mode : {'linear', 'skip', 'norm'}
        linear: input will be multiplied by a biased matrix
        norm: same as linear, but batch norm will be added for input connection
        skip: No operation is performed on the input.  The size must
        match the hidden size.
        (CuDNN docs: cudnnRNNInputMode_t)

    Example
    -------
    >>> import numpy as np
    >>> from odin import backend as K, nnet as N
    >>> def random(*shape):
    ...     return np.random.rand(*shape).astype(CONFIG['floatX']) / 12
    >>> W_in_to_ingate = random(28, 32) / 12
    >>> W_hid_to_ingate = random(32, 32) / 12
    >>> b_ingate = random(32) / 12
    >>> W_in_to_forgetgate = random(28, 32) / 12
    >>> W_hid_to_forgetgate = random(32, 32) / 12
    >>> b_forgetgate = random(32) / 12
    >>> W_in_to_cell = random(28, 32) / 12
    >>> W_hid_to_cell = random(32, 32) / 12
    >>> b_cell = random(32) / 12
    >>> W_in_to_outgate = random(28, 32) / 12
    >>> W_hid_to_outgate = random(32, 32) / 12
    >>> b_outgate = random(32) / 12
    >>> W_cell_to_ingate = random(32) / 12
    >>> W_cell_to_forgetgate = random(32) / 12
    >>> W_cell_to_outgate = random(32) / 12
    >>> cell_init = random(1, 32) / 12
    >>> hid_init = random(1, 32) / 12
    >>> # ====== pre-define parameters ====== #
    >>> x = random(12, 28, 28)
    >>> x_mask = np.random.randint(0, 2, size=(12, 28))
    >>> # x_mask = np.ones(shape=(12, 28))
    >>> # ====== odin ====== #
    >>> X = K.placeholder(shape=(None, 28, 28), name='X')
    >>> mask = K.placeholder(shape=(None, 28), name='mask', dtype='int32')
    >>> f = N.Sequence([
    ...     N.Merge([N.Dense(32, W_init=W_in_to_ingate, b_init=b_ingate, activation=K.linear),
    ...              N.Dense(32, W_init=W_in_to_forgetgate, b_init=b_forgetgate, activation=K.linear),
    ...              N.Dense(32, W_init=W_in_to_cell, b_init=b_cell, activation=K.linear),
    ...              N.Dense(32, W_init=W_in_to_outgate, b_init=b_outgate, activation=K.linear)
    ...             ], merge_function=K.concatenate),
    ...     N.LSTM(32, activation=K.tanh, gate_activation=K.sigmoid,
    ...           W_init=[W_hid_to_ingate, W_hid_to_forgetgate, W_hid_to_cell, W_hid_to_outgate],
    ...           W_peepholes=[W_cell_to_ingate, W_cell_to_forgetgate, W_cell_to_outgate],
    ...           name='lstm')
    >>> ])
    >>> y = f(X, hid_init=hid_init, cell_init=cell_init, mask=mask)
    >>> f = K.function([X, mask], y)
    >>> out1 = f(x, x_mask)[0] # return hidden states

    """

    def __init__(self, num_units,
                 activation=K.tanh,
                 gate_activation=K.sigmoid,
                 W_in_init=K.init.glorot_uniform,
                 W_hid_init=K.init.orthogonal,
                 W_peepholes=K.init.glorot_uniform,
                 b_init=K.init.constant(0.),
                 input_mode='linear',
                 return_cell_memory=False,
                 **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = (K.tanh if activation is None
                           else activation)
        self.gate_activation = (K.sigmoid if gate_activation is None
                                else gate_activation)
        self.W_in_init = W_in_init
        self.W_hid_init = W_hid_init
        self.W_peepholes = W_peepholes
        self.b_init = b_init
        self.input_mode = input_mode
        self.return_cell_memory = return_cell_memory

    @K.rnn_decorator(sequences=['X', 'mask'],
                     states=['h0', 'c0'])
    def _rnn(self, X, h0, c0, mask=None):
        #####################################
        # X: sequences inputs (included bias)
        # init: prev_states
        # W: concatenated [W_update, W_reset]
        # mask: mask inputs (optional)
        prev_states = h0
        prev_memory = c0
        nb_units = self.num_units
        # hidden to hidden connection
        bias = 0 if self.b_init is None else self.b
        _ = X + K.dot(prev_states, self.W_hid) + bias
        hid_input = _slice_x(_, slice(None, nb_units))
        hid_forget = _slice_x(_, slice(nb_units, nb_units * 2))
        hid_hidden = _slice_x(_, slice(nb_units * 2, nb_units * 3))
        hid_output = _slice_x(_, slice(nb_units * 3, None))
        # peepholes connection
        if hasattr(self, 'peepholes'):
            hid_input += prev_memory * _slice_x(self.peepholes,
                                                slice(None, nb_units))
            hid_forget += prev_memory * _slice_x(self.peepholes,
                                                 slice(nb_units, nb_units * 2))

        # calculate new gates
        input_gate = self.gate_activation(hid_input)
        forget_gate = self.gate_activation(hid_forget)
        new_memory = self.activation(hid_hidden)
        # next cell memory
        next_memory = (forget_gate * prev_memory +
                       input_gate * new_memory)
        # output gate
        if hasattr(self, 'peepholes'):
            hid_output += next_memory * _slice_x(self.peepholes,
                                                 slice(nb_units * 2, None))
        output_gate = self.gate_activation(hid_output)
        # new hidden state
        next_states = output_gate * self.activation(next_memory)
        # mask the next state
        if mask is not None:
            next_states = K.switch(mask, next_states, prev_states)
            next_memory = K.switch(mask, next_memory, prev_memory)
        return next_states, next_memory

    def _apply(self, X, h0=None, c0=None, mask=None, **kwargs):
        # check input_shape
        input_shape = K.get_shape(X)
        # ====== check mask ====== #
        if mask is not None and (K.ndim(mask) != 2 or
                                 K.get_shape(mask)[-1] != input_shape[1]):
            raise Exception('Mask must be a 2-D matrix and the time dimension '
                            '(i.e. the second dimension) must equal to "%d"'
                            ', but the given mask has shape "%s".' %
                            (input_shape[1], K.get_shape(mask)))
        # add broadcastable dimension for mask
        if mask is not None:
            mask = K.expand_dims(mask, dim=-1)
        # ====== initialize states ====== #
        # hidden states
        h0 = _check_rnn_hidden_states(h0, self, input_shape, 'h0')
        c0 = _check_rnn_hidden_states(c0, self, input_shape, 'c0')
        # turn off repeat_states if batch_size already included
        if K.get_shape(h0)[0] != 1 and K.get_shape(c0)[0] != 1:
            self.repeat_states = False
        # ====== precompute input ====== #
        # linear or norm input mode
        if self.input_mode != 'skip':
            X = K.dot(X, self.W_in)
            if self.input_mode == 'norm':
                # normalize all axes except the time dimension
                bn = BatchNorm(axes=(0, 1), activation=K.linear,
                               gamma_init=self.gamma, beta_init=self.beta,
                               mean_init=self.mean, inv_std_init=self.inv_std)
                X = bn(X)
        # skip input
        elif input_shape[-1] == self.num_units:
            X = K.repeat(X, 4, axes=-1)
        # ====== compute recurrent output ====== #
        out = self._rnn(X, h0=h0, c0=c0, mask=mask,
                        **self.get_recurrent_info(kwargs))
        if not self.return_cell_memory:
            out = out[:-1]
        for i in out:
            K.add_shape(i, shape=input_shape[:-1] + (self.num_units,))
        # only care about the first state
        return out[0] if len(out) == 1 else out

    def _initialize(self, X):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== check input ====== #
        _init_input2hidden(self, config, rnn_mode='lstm',
                           input_mode=self.input_mode,
                           W_init=self.W_in_init,
                           input_dims=input_shape[-1],
                           hidden_dims=self.num_units)
        # ====== initialize inner parameters ====== #
        # W_input, W_forget, W_hidden, W_output
        config.create_params(self.W_hid_init,
                             shape=(self.num_units, self.num_units),
                             name='W_hid',
                             nnops=self,
                             roles=WEIGHT,
                             nb_params=4)
        # W_input, W_forget, W_output (peepholes is diagonal matrix)
        if self.W_peepholes is not None:
            config.create_params(self.W_peepholes,
                                 shape=(self.num_units,),
                                 name='peepholes',
                                 nnops=self,
                                 roles=WEIGHT,
                                 nb_params=3)
        # bias
        if self.b_init is not None:
            config.create_params(self.b_init,
                                 shape=(self.num_units,),
                                 name='b',
                                 nnops=self,
                                 roles=BIAS,
                                 nb_params=4)
        return config


# ===========================================================================
# DNN
# ===========================================================================
def _check_cudnn_hidden_init(s0, shape, nnops, name):
    nb_layers, batch_size, hidden_size = shape
    # ====== init s0 ====== #
    if s0 is None and hasattr(nnops, name):
        s0 = getattr(nnops, name)
    elif s0 is not None:
        if callable(s0) or K.is_trainable_variable(s0) or isinstance(s0, np.ndarray):
            _ = (nb_layers, 1, hidden_size) if callable(s0) or isinstance(s0, np.ndarray) \
                else K.get_shape(s0)
            s0 = nnops.configuration.create_params(s0, shape=_, name=name,
                                                   nnops=nnops,
                                                   roles=INITIAL_STATE)
        # ====== check s0 shape ====== #
        init_shape = K.get_shape(s0)
        if K.ndim(s0) == 2:
            if K.get_shape(s0)[-1] != hidden_size:
                raise ValueError('init state has %d dimension, but the hidden_size=%d' %
                                (init_shape[-1], hidden_size))
        elif init_shape[::2] != (nb_layers, hidden_size):
            raise ValueError('Require init states of size: %s, but '
                             'given state of size: %s' % (shape, init_shape))
        # ====== return the right shape ====== #
        setattr(nnops, name, s0)
    return s0


class CudnnRNN(NNOps):

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
        linear: input will be multiplied by a biased matrix
        norm: same as linear, but batch norm will be added for input connection
        skip: No operation is performed on the input.  The size must
        match the hidden size.
        (CuDNN docs: cudnnRNNInputMode_t)
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

    def __init__(self, num_units,
            W_init=K.init.glorot_uniform,
            b_init=K.init.constant(0.),
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

    def _initialize(self, x):
        input_shape = K.get_shape(x)
        config = NNConfig(input_shape=input_shape[1:])
        is_bidirectional = self.direction_mode == 'bidirectional'
        # ====== check input ====== #
        if self.input_mode == 'norm':
            _init_input2hidden(self, config, rnn_mode=self.rnn_mode,
                               input_mode=self.input_mode,
                               W_init=self.W_init,
                               input_dims=input_shape[-1],
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
            with K.variable_scope(self.name):
                parameters = [init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                        W_init=self.W_init, b_init=self.b_init,
                                        one_vector=False, return_variable=True,
                                        bidirectional=is_bidirectional,
                                        name='layer%d' % i)
                              for i in range(self.num_layers)]
            # print([(j.name, j.tag.roles) for i in parameters for j in i]); exit()
            for p in chain(*parameters):
                config.create_params(p, shape=K.get_shape(p),
                                     name=p.name.split(':')[0].split('/')[1],
                                     nnops=self, roles=PARAMETER)
        # else initialize all in 1 big vector
        else:
            parameters = np.concatenate([init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                         one_vector=True, return_variable=False,
                                         bidirectional=is_bidirectional)
                                         for i in range(self.num_layers)])
            config.create_params(parameters, shape=parameters.shape,
                                 name='params', nnops=self, roles=PARAMETER)
        return config

    def _apply(self, X, h0=None, c0=None, mask=None):
        batch_size = K.get_shape(X, native=True)[0]
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
            shapeX = K.get_shape(X, native=True)
            ndims = K.ndim(X)
            if 'rnn' in self.rnn_mode: N = 1
            elif self.rnn_mode == 'gru': N = 3
            else: N = 4
            newshape = [shapeX[i] for i in range(ndims - 1)] + [self.num_units, N]
            X = K.mean(K.reshape(X, newshape), axis=-1)
        # ====== hidden state ====== #
        num_layers = self.num_layers * 2 if is_bidirectional else self.num_layers
        require_shape = (num_layers, batch_size, self.num_units)
        h0 = _check_cudnn_hidden_init(h0, require_shape, self, 'h0')
        c0 = _check_cudnn_hidden_init(c0, require_shape, self, 'c0')
        # ====== parameters ====== #
        if self.params_split:
            parameters = K.concatenate([K.flatten(i, outdim=1)
                                        for i in self.parameters
                                        if not has_roles(i, INITIAL_STATE)])
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


# ===========================================================================
# Auto RNN
# ===========================================================================
def AutoRNN(num_units, W_init=K.init.glorot_uniform, b_init=K.init.constant(0.),
            rnn_mode='lstm', num_layers=1,
            input_mode='linear',
            direction_mode='unidirectional',
            params_split=False,
            return_states=False,
            dropout=0., name=None, prefer_cudnn=True):
    """ Automatically select best RNN implementation (using Cudnn RNN
    if available).

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
    input_mode : {'linear', 'skip'}
        linear: input will be multiplied by a biased matrix
        skip: No operation is performed on the input.  The size must
        match the hidden size.
        (CuDNN docs: cudnnRNNInputMode_t)
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

    output_shape: (batch_size, timesteps, num_units)
    hidden_shape: (num_layers, batch_size, num_units)
    cell_shape: (num_layers, batch_size, num_units)

    """
    # ====== using cudnn ====== #
    if prefer_cudnn and K.cudnn_available():
        return CudnnRNN(num_units=num_units,
            W_init=W_init, b_init=b_init,
            rnn_mode=rnn_mode, num_layers=num_layers,
            input_mode=input_mode,
            direction_mode=direction_mode,
            params_split=params_split,
            return_states=return_states,
            dropout=dropout, name=name)
    # ====== using scan ====== #
    else:
        layers = []
        for i in range(num_layers):
            creator = None
            if 'rnn' in rnn_mode:
                kwargs = {'num_units':num_units,
                          'W_init':W_init,
                          'b_init':b_init,
                          'input_mode':input_mode}
                if 'relu' in rnn_mode:
                    kwargs['activation'] = K.relu
                else:
                    kwargs['activation'] = K.tanh
                creator = RNN
            elif rnn_mode == 'gru':
                kwargs = {'num_units':num_units,
                 'activation':K.tanh,
                 'gate_activation':K.sigmoid,
                 'W_in_init':W_init,
                 'W_hid_init':W_init,
                 'b_init':b_init,
                 'input_mode':input_mode}
                creator = GRU
            elif rnn_mode == 'lstm':
                kwargs = {'num_units':num_units,
                 'activation':K.tanh,
                 'gate_activation':K.sigmoid,
                 'W_in_init':W_init,
                 'W_hid_init':W_init,
                 'W_peepholes':W_init,
                 'b_init':b_init,
                 'input_mode':input_mode,
                 'return_cell_memory':return_states}
                creator = LSTM
            # direction mode
            layer_name = name + '_%d' % i if name is not None else None
            if direction_mode == 'unidirectional':
                layers.append(creator(backwards=False, name=layer_name, **kwargs))
            else:
                layers.append(
                    BidirectionalRNN(
                        forward=creator(backwards=False, name=layer_name, **kwargs),
                        mode='concat')
                )
        return Sequence(layers, debug=False)
