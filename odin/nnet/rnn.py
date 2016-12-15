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
    forward: p
        p
    backward: p
        p
    mode: callable
        callable:
    """

    def __init__(self, forward, backward=None, mode=K.concatenate, **kwargs):
        if not isinstance(forward, BaseRNN):
            raise ValueError('forward must be instance of BaseRNN, but it is %s'
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
        if h0 is None and hasattr(self, 'h0'):
            h0 = self.h0
        else:
            h0 = K.init.constant(0.) if h0 is None else h0
            # only store trainable variable or constant
            if callable(h0) or K.is_trainable_variable(h0):
                h0 = self.configuration.create_params(h0,
                    shape=(1,) + input_shape[2:-1] + (self.num_units,),
                    name='h0',
                    nnops=self,
                    roles=INITIAL_STATE)
            else:
                self.h0 = h0
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
        # input connection
        if self.input_mode != 'skip':
            config.create_params(W_init[0],
                                 shape=(input_shape[-1], self.num_units),
                                 name='W_in',
                                 nnops=self,
                                 roles=WEIGHT)
            if self.input_mode == 'norm':
                config.create_params(K.init.constant(0.), shape=(self.num_units,),
                                     name='beta',
                                     nnops=self, roles=BATCH_NORM_SHIFT_PARAMETER)
                config.create_params(K.init.constant(1.), shape=(self.num_units,),
                                     name='gamma',
                                     nnops=self, roles=BATCH_NORM_SCALE_PARAMETER)
                config.create_params(K.init.constant(0.), shape=(self.num_units,),
                                     name='mean',
                                     nnops=self, roles=BATCH_NORM_POPULATION_MEAN)
                config.create_params(K.init.constant(1.), shape=(self.num_units,),
                                     name='inv_std',
                                     nnops=self, roles=BATCH_NORM_POPULATION_INVSTD)
        # skip input mode
        elif input_shape[-1] != self.num_units:
            raise Exception('Skip input mode, input trailing_dimension=%d '
                            '(the final dim) must equal to the number of '
                            'hidden unit, which is: %d' %
                            (input_shape[-1], self.num_units))
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
    W_init: variable, ndarray or callable (shape: (num_units, num_units))
        Initializer for hidden-to-hidden weight matrices, if a list is given,
        the weights will be initialized in following order:
        "[W_hid_to_updategate, W_hid_to_resetgate, W_hid_to_hidden_update]"
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
        if h0 is None and hasattr(self, 'h0'):
            h0 = self.h0
        else:
            h0 = K.init.constant(0.) if h0 is None else h0
            # only store trainable variable or constant
            if callable(h0) or K.is_trainable_variable(h0):
                h0 = self.configuration.create_params(h0,
                    shape=(1,) + input_shape[2:-1] + (self.num_units,),
                    name='h0',
                    nnops=self,
                    roles=INITIAL_STATE)
            else:
                self.h0 = h0
        # turn off repeat_states if batch_size already included
        if K.get_shape(h0)[0] != 1:
            self.repeat_sxtates = False
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
        if self.input_mode != 'skip':
            config.create_params(self.W_in_init,
                                 shape=(input_shape[-1], self.num_units),
                                 name='W_in',
                                 nnops=self,
                                 roles=WEIGHT,
                                 nb_params=3)
            if self.input_mode == 'norm':
                config.create_params(K.init.constant(0.), shape=(self.num_units * 3,),
                                     name='beta',
                                     nnops=self, roles=BATCH_NORM_SHIFT_PARAMETER)
                config.create_params(K.init.constant(1.), shape=(self.num_units * 3,),
                                     name='gamma',
                                     nnops=self, roles=BATCH_NORM_SCALE_PARAMETER)
                config.create_params(K.init.constant(0.), shape=(self.num_units * 3,),
                                     name='mean',
                                     nnops=self, roles=BATCH_NORM_POPULATION_MEAN)
                config.create_params(K.init.constant(1.), shape=(self.num_units * 3,),
                                     name='inv_std',
                                     nnops=self, roles=BATCH_NORM_POPULATION_INVSTD)
        elif input_shape[-1] != self.num_units and \
        input_shape[-1] != self.num_units * 3:
            raise Exception('Skip input mode, Input trailing_dimension=%d '
                            '(the final dim) must equal to the number of hidden '
                            'units (tied input connection), or triple the number '
                            'of hidden units (1 for W_update, 1 for W_reset, '
                            'and 1 for W_hidden) which is: %d' %
                            (input_shape[-1], self.num_units * 3))
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

    Parameters
    ----------
    num_units: int
        pass
    activation: callable
        activation for hidden state
    gate_activation: callable
        activation for each gate
    W_init: variable, ndarray or callable (shape: (num_units, num_units))
        Initializer for hidden-to-hidden weight matrices, if a list is given,
        the weights will be initialized in following order:
        "[W_hid_to_inputgate, W_hid_to_forgetgate, W_hid_to_cell,
          W_hid_to_outputgate]"
    W_peepholes: variable, ndarray or callable (shape: (num_units,))
        if `W_peepholes=None`, no peepholes are introduced. If a list is given,
        the weights will be initialized in following order:
        "[W_cell_to_inputgate, W_hid_to_forgetgate, W_hid_to_outputgate]""


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
        if h0 is None and hasattr(self, 'h0'):
            h0 = self.h0
        else:
            h0 = K.init.constant(0.) if h0 is None else h0
            if callable(h0) or K.is_trainable_variable(h0):
                h0 = self.configuration.create_params(h0,
                    shape=(1,) + input_shape[2:-1] + (self.num_units,),
                    name='h0',
                    nnops=self,
                    roles=INITIAL_STATE)
            else:
                self.h0 = h0
        # memory
        if c0 is None and hasattr(self, 'c0'):
            c0 = self.c0
        else:
            c0 = K.init.constant(0.) if c0 is None else c0
            if callable(c0) or K.is_trainable_variable(c0):
                c0 = self.configuration.create_params(c0,
                    shape=(1,) + input_shape[2:-1] + (self.num_units,),
                    name='c0',
                    nnops=self,
                    roles=INITIAL_STATE)
            else:
                self.c0 = c0
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
        if self.input_mode != 'skip':
            config.create_params(self.W_in_init,
                                 shape=(input_shape[-1], self.num_units),
                                 name='W_in',
                                 nnops=self,
                                 roles=WEIGHT,
                                 nb_params=4)
            if self.input_mode == 'norm':
                config.create_params(K.init.constant(0.), shape=(self.num_units * 4,),
                                     name='beta',
                                     nnops=self, roles=BATCH_NORM_SHIFT_PARAMETER)
                config.create_params(K.init.constant(1.), shape=(self.num_units * 4,),
                                     name='gamma',
                                     nnops=self, roles=BATCH_NORM_SCALE_PARAMETER)
                config.create_params(K.init.constant(0.), shape=(self.num_units * 4,),
                                     name='mean',
                                     nnops=self, roles=BATCH_NORM_POPULATION_MEAN)
                config.create_params(K.init.constant(1.), shape=(self.num_units * 4,),
                                     name='inv_std',
                                     nnops=self, roles=BATCH_NORM_POPULATION_INVSTD)
        # skip input mode
        elif input_shape[-1] != self.num_units and \
        input_shape[-1] != self.num_units * 4: # 3 gates + 1 hid_update
            raise Exception('Skip input mode, input trailing_dimension=%d '
                            '(the final dim) must equal to the number of hidden '
                            'units (tied input connection), or 4-th the number '
                            'of hidden units (1 for W_input, 1 for W_forget, '
                            '1 for W_hidden, and 1 for W_output), which is: %d' %
                            (input_shape[-1], self.num_units * 4))
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
class CudnnRNN(NNOps):

    """CuDNN v5 RNN implementation.

    Parameters
    ----------
    hidden_size : int
        the number of units within the RNN model.
    W_init:
        initial description for weights
    b_init:
        initial description for bias
    initial_states: list of tensor
        h0 with shape [num_layers, batch_size, hidden_size]
        c0 (lstm) with shape [num_layers, batch_size, hidden_size]
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

    output_shape: (batch_size, timesteps, hidden_size)
    hidden_shape: (num_layers, batch_size, hidden_size)
    cell_shape: (num_layers, batch_size, hidden_size)

    """

    def __init__(self, hidden_size,
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
        self.hidden_size = hidden_size
        self.rnn_mode = rnn_mode
        self.num_layers = num_layers
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

    def _initialize(self, x, hid_init=None, cell_init=None):
        input_shape = K.get_shape(x)
        config = NNConfig(input_shape=input_shape[1:])
        is_bidirectional = self.direction_mode == 'bidirectional'
        # ====== create params ====== #
        layer_info = [input_shape[-1], self.hidden_size] + \
                     [self.hidden_size * (2 if is_bidirectional else 1),
                      self.hidden_size] * (self.num_layers - 1)
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
                                     nnops=self)
        # else initialize all in 1 big vector
        else:
            parameters = np.concatenate([init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                         one_vector=True, return_variable=False,
                                         bidirectional=True if is_bidirectional else False)
                                         for i in range(self.num_layers)])
            config.create_params(parameters, shape=parameters.shape,
                                 name='params', nnops=self)
        # ====== create initials states ====== #
        if self.hid_init is not None:
            num_layers = self.num_layers * 2 if is_bidirectional else self.num_layers
            batch_size = 1 if input_shape[0] is None else input_shape[0]
            require_shape = (num_layers, batch_size, self.hidden_size)
            # check hidden states
            h0 = hid_init
            check_shape = require_shape
            if K.is_variable(h0) or isinstance(h0, np.ndarray):
                if K.get_shape(h0)[::2] != (num_layers, self.hidden_size):
                    raise ValueError('Require hidden_state of size: %s, but '
                                     'given state of size: %s' %
                                     (require_shape, K.get_shape(h0)))
                check_shape = K.get_shape(h0)
            config.create_params(h0, shape=check_shape, name='h0',
                                 nnops=self, roles=INITIAL_STATE)
        # do the same for cell states
        if self.rnn_mode == 'lstm' and cell_init is not None:
            c0 = cell_init
            check_shape = require_shape
            if K.is_variable(c0) or isinstance(c0, np.ndarray):
                if K.get_shape(c0)[::2] != (num_layers, self.hidden_size):
                    raise ValueError('Require cell_state of size: %s, but '
                                     'given state of size: %s' %
                                     (require_shape, K.get_shape(c0)))
                check_shape = K.get_shape(c0)
            config.create_params(c0, shape=check_shape, name='c0',
                                 nnops=self, roles=INITIAL_STATE)
        return config

    def _apply(self, x, hid_init=None, cell_init=None):
        batch_size = K.get_shape(x, native=True)[0]
        # ====== hidden state ====== #
        initial_states = None
        if hasattr(self, 'h0'):
            h0 = self.h0
            if K.get_shape(self.h0)[1] is 1:
                h0 = K.repeat(h0, batch_size, axes=1)
            initial_states = [h0]
        # cell state for lstm
        if self.rnn_mode == 'lstm':
            if hasattr(self, 'c0'):
                c0 = self.c0
                if K.get_shape(c0)[1] is 1:
                    c0 = K.repeat(c0, batch_size, axes=1)
                initial_states.append(c0)
            else:
                initial_states.append(None)
        # ====== parameters ====== #
        if self.params_split:
            parameters = K.concatenate([K.flatten(i, outdim=1)
                                        for i in self.parameters
                                        if not has_roles(i, INITIAL_STATE)])
        else:
            parameters = self.params
        # ====== return CuDNN RNN ====== #
        results = K.rnn_dnn(x, hidden_size=self.hidden_size, rnn_mode=self.rnn_mode,
                           num_layers=self.num_layers,
                           initial_states=initial_states,
                           parameters=parameters,
                           input_mode=self.input_mode,
                           direction_mode=self.direction_mode,
                           dropout=self.dropout, name=self.name)
        if not self.return_states:
            results = results[0] # only get the output
        return results


# ===========================================================================
# Auto RNN
# ===========================================================================
def AutoRNN(hidden_size, W_init=K.init.glorot_uniform, b_init=K.init.constant(0.),
            initial_states=None,
            rnn_mode='lstm', num_layers=1,
            input_mode='linear',
            direction_mode='unidirectional',
            params_split=False,
            return_states=False,
            dropout=0., name=None):
    """ Automatically select best RNN implementation (using Cudnn RNN
    if available).

    Parameters
    ----------
    hidden_size : int
        the number of units within the RNN model.
    W_init:
        initial description for weights
    b_init:
        initial description for bias
    initial_states: list of tensor
        h0 with shape [num_layers, batch_size, hidden_size]
        c0 (lstm) with shape [num_layers, batch_size, hidden_size]
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

    output_shape: (batch_size, timesteps, hidden_size)
    hidden_shape: (num_layers, batch_size, hidden_size)
    cell_shape: (num_layers, batch_size, hidden_size)

    """
    # ====== using cudnn ====== #
    if K.cudnn_available():
        if input_mode == 'norm':
            input_mode = 'linear'
        return CudnnRNN(hidden_size=hidden_size,
            W_init=W_init, b_init=b_init,
            initial_states=initial_states,
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
                kwargs = {'num_units':hidden_size,
                          'W_init':W_init,
                          'b_init':b_init,
                          'input_mode':input_mode,
                          'name':name}
                if 'relu' in rnn_mode:
                    kwargs['activation'] = K.relu
                else:
                    kwargs['activation'] = K.tanh
                creator = RNN
            elif rnn_mode == 'gru':
                kwargs = {'num_units':hidden_size,
                 'activation':K.tanh,
                 'gate_activation':K.sigmoid,
                 'W_in_init':W_init,
                 'W_hid_init':W_init,
                 'b_init':b_init,
                 'input_mode':input_mode,
                 'name':name}
                creator = GRU
            elif rnn_mode == 'lstm':
                kwargs = {'num_units':hidden_size,
                 'activation':K.tanh,
                 'gate_activation':K.sigmoid,
                 'W_in_init':W_init,
                 'W_hid_init':W_init,
                 'W_peepholes':W_init,
                 'b_init':b_init,
                 'input_mode':input_mode,
                 'return_cell_memory':return_states,
                 'name':name}
                creator = LSTM
            # direction mode
            if direction_mode == 'unidirectional':
                layers.append(creator(backwards=False, **kwargs))
            else:
                layers.append(
                    BidirectionalRNN(forward=creator(backwards=False, **kwargs),
                        mode='concat')
                )
        return Sequence(layers, debug=True)
