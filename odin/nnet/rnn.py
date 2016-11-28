from __future__ import division, absolute_import, print_function

from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np

from odin import backend as K
from odin.basic import INITIAL_STATE, WEIGHT, BIAS, PARAMETER
from odin.utils import as_tuple

from .base import NNConfig, NNOps


# ===========================================================================
# Helper
# ===========================================================================
@add_metaclass(ABCMeta)
class BaseRNN(NNOps):

    def __init__(self, **kwargs):
        super(BaseRNN, self).__init__(**kwargs)
        # ====== defaults recurrent control ====== #
        self.repeat_states = True
        self.iterate = True
        self.backwards = False
        self.n_steps = None
        self.batch_size = None

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


def _slice_x(X, idx):
    """ Slice tensor at its last dimension """
    ndim = K.ndim(X)
    _ = [slice(None, None, None) for i in range(ndim - 1)]
    return X[_ + [idx]]


# ===========================================================================
# Generalized RNN
# ===========================================================================
class GeneralizedRNN(BaseRNN):
    """ GeneralizedRNN """

    def __init__(self, **kwargs):
        super(GeneralizedRNN, self).__init__(**kwargs)


# ===========================================================================
# RNN
# ===========================================================================
class SimpleRecurrent(BaseRNN):
    """
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
    ...     N.SimpleRecurrent(num_units=32, activation=K.relu,
    ...         W_init=W[1])
    >>> ])
    >>> return X1, f(X1, hid_init=zeros(1, 32))[0]

    """

    def __init__(self, num_units, activation=K.relu,
                 W_init=K.init.glorot_uniform, **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = K.linear if activation is None else activation
        self.W_init = W_init

    @K.rnn_decorator(sequences=['X', 'mask'], states=['hid_init'])
    def _rnn(self, X, hid_init, mask=None):
        next_states = self.activation(X + K.dot(hid_init, self.W))
        if mask is not None:
            next_states = K.switch(mask, next_states, hid_init)
        return next_states

    def _apply(self, X, hid_init=None, mask=None, **kwargs):
        input_shape = K.get_shape(X)
        out = self._rnn(X, hid_init=self.hid_init, mask=mask,
                        **self.get_recurrent_info(kwargs))
        for i in out:
            K.add_shape(i, shape=input_shape)
        # only care about the first state
        return out[0] if len(out) == 1 else out

    def _initialize(self, X, hid_init=None, mask=None):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== check input ====== #
        input_shape = K.get_shape(X)
        if input_shape[-1] != self.num_units:
            raise Exception('Input trailing_dimension=%d (the final dim) must '
                            'equal to the number of hidden unit, '
                            'which is: %d' % (input_shape[-1], self.num_units))
        # ====== initialize states ====== #
        if hid_init is None:
            hid_init = K.init.constant(0.)
        # create init of state
        state_init = config.create_params(hid_init,
            shape=(1,) + input_shape[2:-1] + (self.num_units,),
            name='hid_init',
            nnops=self,
            roles=INITIAL_STATE)
        # turn off repeat_states if batch_size already included
        if K.get_shape(state_init)[0] != 1:
            self.repeat_states = False
        # ====== check mask ====== #
        if mask is not None and (K.ndim(mask) != K.ndim(X) - 1 or
                                 K.get_shape(mask)[-1] != input_shape[1]):
            raise Exception('Mask must has "%d" dimensions and the time dimension '
                            '(i.e. the second dimension) must equal to "%d"'
                            ', but the given mask has shape "%s".' %
                            (K.ndim(X) - 1, input_shape[1], K.get_shape(mask)))
        # ====== initialize inner parameters ====== #
        config.create_params(self.W_init,
                             shape=(self.num_units, self.num_units),
                             name='W',
                             nnops=self,
                             roles=WEIGHT)
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
                 W_init=K.init.glorot_normal,
                 **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = (K.tanh if activation is None
                           else activation)
        self.gate_activation = (K.sigmoid if gate_activation is None
                                else gate_activation)
        self.W_init = W_init

    @K.rnn_decorator(sequences=['X', 'mask'], states=['hid_init'])
    def _rnn(self, X, hid_init, mask=None):
        #####################################
        # X: sequences inputs (included bias)
        # init: prev_states
        # W: concatenated [W_update, W_reset]
        # mask: mask inputs (optional)
        prev_states = hid_init
        nb_units = self.num_units
        # hidden connection of all gates and states update
        hid_connection = K.dot(prev_states, self.W)
        # hidden to hidden connection
        hid_gate = _slice_x(hid_connection, slice(None, nb_units * 2))
        X_gate = _slice_x(X, slice(None, nb_units * 2))
        hid_states = _slice_x(hid_connection, slice(nb_units * 2, None))
        X_states = _slice_x(X, slice(nb_units * 2, None))
        # new gates
        _ = self.gate_activation(X_gate + hid_gate)
        update_values = _slice_x(_, slice(None, nb_units))
        reset_values = _slice_x(_, slice(nb_units, nb_units * 2))
        # calculate new gates
        new_states = self.activation(X_states + reset_values * hid_states)
        # final new states
        next_states = (new_states * update_values +
                       prev_states * (1 - update_values))
        # mask the next state
        if mask is not None:
            next_states = K.switch(mask, next_states, prev_states)
        return next_states

    def _apply(self, X, hid_init=None, mask=None, **kwargs):
        # check input_shape
        input_shape = K.get_shape(X)
        if input_shape[-1] == self.num_units:
            X = K.repeat(X, 3, axes=-1)
        # add broadcastable dimension for mask
        if mask is not None:
            mask = K.expand_dims(mask, dim=-1)
        # recurrent
        out = self._rnn(X, hid_init=self.hid_init, mask=mask,
                        **self.get_recurrent_info(kwargs))
        for i in out:
            K.add_shape(i, shape=input_shape)
        # only care about the first state
        return out[0] if len(out) == 1 else out

    def _initialize(self, X, hid_init=None, mask=None):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== check input ====== #
        if input_shape[-1] != self.num_units and \
        input_shape[-1] != self.num_units * 3:
            raise Exception('Input trailing_dimension=%d (the final dim) must '
                            'equal to the number of hidden units (tied input '
                            'connection), or triple the number of hidden units'
                            '(1 for W_update, 1 for W_reset, and 1 for W_hidden) '
                            'which is: %d' % (input_shape[-1], self.num_units * 3))
        # ====== initialize states ====== #
        if hid_init is None:
            hid_init = K.init.constant(0)
        # create init of state
        state_init = config.create_params(hid_init,
            shape=(1,) + input_shape[2:-1] + (self.num_units,),
            name='hid_init',
            nnops=self,
            roles=INITIAL_STATE)
        # turn off repeat_states if batch_size already included
        if K.get_shape(state_init)[0] != 1:
            self.repeat_states = False
        # ====== check mask ====== #
        if mask is not None and (K.ndim(mask) != 2 or
                                 K.get_shape(mask)[-1] != input_shape[1]):
            raise Exception('Mask must be a 2-D matrix and the time dimension '
                            '(i.e. the second dimension) must equal to "%d"'
                            ', but the given mask has shape "%s".' %
                            (input_shape[1], K.get_shape(mask)))
        # ====== initialize inner parameters ====== #
        # W_update, W_reset, W_hidden
        config.create_params(self.W_init,
                             shape=(self.num_units, self.num_units),
                             name='W',
                             nnops=self,
                             roles=WEIGHT,
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
                 W_init=K.init.orthogonal,
                 W_peepholes=K.init.glorot_uniform,
                 return_cell_memory=False,
                 **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = (K.tanh if activation is None
                           else activation)
        self.gate_activation = (K.sigmoid if gate_activation is None
                                else gate_activation)
        self.W_init = W_init
        self.W_peepholes = W_peepholes
        self.return_cell_memory = return_cell_memory

    @K.rnn_decorator(sequences=['X', 'mask'],
                     states=['hid_init', 'cell_init'])
    def _rnn(self, X, hid_init, cell_init, tied_input, mask=None):
        #####################################
        # X: sequences inputs (included bias)
        # init: prev_states
        # W: concatenated [W_update, W_reset]
        # mask: mask inputs (optional)
        prev_states = hid_init
        prev_memory = cell_init
        nb_units = self.num_units
        # hidden to hidden connection
        _ = X + K.dot(prev_states, self.W)
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

    def _apply(self, X, hid_init=None, cell_init=None, mask=None, **kwargs):
        # check input_shape
        input_shape = K.get_shape(X)
        tied_input = False
        if input_shape[-1] == self.num_units:
            X = K.repeat(X, 4, axes=-1)
        # add broadcastable dimension for mask
        if mask is not None:
            mask = K.expand_dims(mask, dim=-1)
        # recurrent
        out = self._rnn(X, hid_init=self.hid_init, cell_init=self.cell_init,
                        tied_input=tied_input, mask=mask,
                        **self.get_recurrent_info(kwargs))
        if not self.return_cell_memory:
            out = out[:-1]
        for i in out:
            K.add_shape(i, shape=input_shape[:-1] + (self.num_units,))
        # only care about the first state
        return out[0] if len(out) == 1 else out

    def _initialize(self, X, hid_init=None, cell_init=None, mask=None):
        input_shape = K.get_shape(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== check input ====== #
        if input_shape[-1] != self.num_units and \
        input_shape[-1] != self.num_units * 4: # 3 gates + 1 hid_update
            raise Exception('Input trailing_dimension=%d (the final dim) must '
                            'equal to the number of hidden units (tied input '
                            'connection), or 4-th the number of hidden units'
                            '(1 for W_input, 1 for W_forget, 1 for W_hidden, and '
                            '1 for W_output), which is: %d' %
                            (input_shape[-1], self.num_units * 4))
        # ====== initialize states ====== #
        if hid_init is None:
            hid_init = K.init.constant(0.)
        if cell_init is None:
            cell_init = K.init.constant(0.)
        # create init of state
        hid_init = config.create_params(hid_init,
            shape=(1,) + input_shape[2:-1] + (self.num_units,),
            name='hid_init',
            nnops=self,
            roles=INITIAL_STATE)
        cell_init = config.create_params(cell_init,
            shape=(1,) + input_shape[2:-1] + (self.num_units,),
            name='cell_init',
            nnops=self,
            roles=INITIAL_STATE)
        # turn off repeat_states if batch_size already included
        if not (K.get_shape(hid_init)[0] == 1 and
                K.get_shape(cell_init)[0] == 1):
            self.repeat_states = False
        # ====== check mask ====== #
        if mask is not None and (K.ndim(mask) != 2 or
                                 K.get_shape(mask)[-1] != input_shape[1]):
            raise Exception('Mask must be a 2-D matrix and the time dimension '
                            '(i.e. the second dimension) must equal to "%d"'
                            ', but the given mask has shape "%s".' %
                            (input_shape[1], K.get_shape(mask)))
        # ====== initialize inner parameters ====== #
        # W_input, W_forget, W_hidden, W_output
        config.create_params(self.W_init,
                             shape=(self.num_units, self.num_units),
                             name='W',
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
        return config


# ===========================================================================
# DNN
# ===========================================================================
class CudnnRNN(NNOps):
    """CuDNN v5 RNN implementation.

    Parameters
    ----------
    X : input varialbe or placeholder
        shape=(batch_size, timesteps, input_dims)
    hidden_size : int
        the number of units within the RNN model.
    rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
        See cudnn documentation for ``cudnnRNNMode_t``.
    num_layers : int
        the number of layers for the RNN model.
    initial_states: list of tensor
        pass
    parameters: list of tensor
        pass
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
            initial_states=K.init.constant(0.),
            rnn_mode='lstm', num_layers=1,
            input_mode='linear',
            direction_mode='unidirectional',
            dropout=0., **kwargs):
        super(CudnnRNN, self).__init__(**kwargs)
        # ====== defaults recurrent control ====== #
        self.hidden_size = hidden_size
        self.rnn_mode = rnn_mode
        self.num_layers = num_layers
        self.input_mode = input_mode
        self.direction_mode = direction_mode
        self.dropout = dropout

        self.initial_states = as_tuple(initial_states, N=2 if rnn_mode == 'lstm' else 1)
        self.W_init = W_init
        self.b_init = b_init

    # ==================== abstract methods ==================== #
    def _transpose(self):
        # flip the input and hidden
        raise NotImplementedError

    def _initialize(self, x):
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
        parameters = np.concatenate([init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                     W_init=self.W_init, b_init=self.b_init,
                                     one_vector=True, return_variable=False,
                                     bidirectional=True if is_bidirectional else False)
                                     for i in range(self.num_layers)]).astype(K.floatX())
        config.create_params(parameters, shape=parameters.shape, name='params',
                             nnops=self, roles=PARAMETER)
        # ====== create initials states ====== #
        num_layers = self.num_layers * 2 if is_bidirectional else self.num_layers
        batch_size = 1 if input_shape[0] is None else input_shape[0]
        config.create_params(self.initial_states[0],
                             shape=(num_layers, batch_size, self.hidden_size),
                             name='h0', nnops=self, roles=INITIAL_STATE)
        if self.rnn_mode == 'lstm':
            config.create_params(self.initial_states[1],
                                 shape=(num_layers, batch_size, self.hidden_size),
                                 name='c0', nnops=self, roles=INITIAL_STATE)
        return config

    def _apply(self, x):
        batch_size = K.get_shape(x, native=True)[0]
        # hidden state
        h0 = self.h0
        if K.get_shape(self.h0)[1] is 1:
            h0 = K.repeat(h0, batch_size, axes=1)
        initial_states = [h0]
        # cell state for lstm
        if self.rnn_mode == 'lstm':
            c0 = self.c0
            if K.get_shape(c0)[1] is 1:
                c0 = K.repeat(c0, batch_size, axes=1)
            initial_states.append(c0)
        # return CuDNN RNN
        return K.rnn_dnn(x, hidden_size=self.hidden_size, rnn_mode=self.rnn_mode,
                         num_layers=self.num_layers,
                         initial_states=initial_states,
                         parameters=self.params,
                         input_mode=self.input_mode,
                         direction_mode=self.direction_mode,
                         dropout=self.dropout, name=self.name)
