from __future__ import division, absolute_import, print_function

from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np

from odin import backend as K
from odin.roles import INITIAL_STATE, WEIGHT, BIAS

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
        self.go_backwards = False
        self.n_steps = None
        self.batch_size = None

    @abstractmethod
    def _rnn(self, **kwargs):
        pass

    def get_recurrent_info(self):
        """ return information that control how this ops recurrently
        performed
        """
        return {
            'iterate': self.iterate,
            'go_backwards': self.go_backwards,
            'repeat_states': self.repeat_states,
            'name': self.name,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
        }


def _slice_x(X, idx):
    """ Slice tensor at its last dimension """
    ndim = K.ndim(X)
    _ = [slice(None, None, None) for i in range(ndim - 1)]
    return X[_ + [idx]]


# ===========================================================================
# RNN
# ===========================================================================
class SimpleRecurrent(BaseRNN):

    def __init__(self, num_units, activation=K.relu,
                 W_init=K.init.glorot_uniform,
                 state_init=K.init.constant(0), **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = K.linear if activation is None else activation
        self.W_init = W_init
        self.state_init = state_init

    @K.rnn_decorator(sequences=['X', 'mask'], states=['init'])
    def _rnn(self, X, init, mask=None):
        next_states = self.activation(X + K.dot(init, self.W))
        if mask is not None:
            next_states = K.switch(mask, next_states, init)
        return next_states

    def _apply(self, X, init=None, mask=None):
        input_shape = K.get_shape(X)
        out = self._rnn(X, init, mask, **self.get_recurrent_info())
        for i in out:
            K.add_shape(i, shape=input_shape)
        # only care about the first state
        return out

    def _initialize(self, X, init=None, mask=None):
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
        if init is None:
            if isinstance(self.state_init, np.ndarray) and \
            self.state_init.ndim == 1:
                self.state_init = self.state_init.reshape(1, -1)
            config.create_params(self.state_init,
                                 shape=(1, self.num_units),
                                 name='init',
                                 nnops=self,
                                 roles=INITIAL_STATE)
        else:
            if K.get_shape(init)[-1] != self.num_units:
                raise Exception('Initialization of state must has trialing '
                                'dimension equal to number of hidden unit, '
                                'which is: %d' % self.num_units)
            self.init = init
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


class GRU(BaseRNN):

    def __init__(self, num_units,
                 activation=K.tanh,
                 gate_activation=K.sigmoid,
                 W_init=K.init.glorot_normal,
                 state_init=K.init.constant(0), **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = (K.tanh if activation is None
                           else activation)
        self.gate_activation = (K.sigmoid if gate_activation is None
                                else gate_activation)
        self.W_init = W_init
        self.state_init = state_init

    @K.rnn_decorator(sequences=['X', 'mask'], states=['init'])
    def _rnn(self, X, init, tied_input, mask=None):
        #####################################
        # X: sequences inputs (included bias)
        # init: prev_states
        # W: concatenated [W_update, W_reset]
        # mask: mask inputs (optional)
        prev_states = init
        nb_units = self.num_units
        # input to hidden connection
        if tied_input:
            X_update = X
            X_reset = X
            X_hidden = X
        else:
            X_update = _slice_x(X, slice(None, nb_units))
            X_reset = _slice_x(X, slice(nb_units, nb_units * 2))
            X_hidden = _slice_x(X, slice(nb_units * 2, None))
        # hidden to hidden connection
        _ = K.dot(prev_states, self.W)
        hid_update = _slice_x(_, slice(None, nb_units))
        hid_reset = _slice_x(_, slice(nb_units, nb_units * 2))
        hid_hidden = _slice_x(_, slice(nb_units * 2, None))
        # calculate new gates
        update_values = self.gate_activation(X_update + hid_update)
        reset_values = self.gate_activation(X_reset + hid_reset)
        new_states = self.activation(X_hidden + reset_values * hid_hidden)
        # final new states
        next_states = (new_states * update_values +
                       prev_states * (1 - update_values))
        # mask the next state
        if mask is not None:
            next_states = K.switch(mask, next_states, prev_states)
        return next_states

    def _apply(self, X, init=None, mask=None):
        # check input_shape
        input_shape = K.get_shape(X)
        tied_input = False
        if input_shape[-1] == self.num_units:
            tied_input = True
        print(mask)
        # recurrent
        out = self._rnn(X, init, tied_input, mask,
                        **self.get_recurrent_info())
        for i in out:
            K.add_shape(i, shape=input_shape)
        # only care about the first state
        return out

    def _initialize(self, X, init=None, mask=None):
        input_shape = K.get_shape(X)
        ndim = K.ndim(X)
        config = NNConfig(input_shape=input_shape,
                          num_units=self.num_units)
        # ====== check input ====== #
        if input_shape[-1] != self.num_units and \
        input_shape[-1] != self.num_units * 3:
            raise Exception('Input trailing_dimension=%d (the final dim) must '
                            'equal to the number of hidden units (tied input '
                            'connection), or triple the number of hidden units'
                            '(1 for W_update, 1 for W_reset, and 1 for W_hidden) '
                            'which is: %d' % (input_shape[-1], self.num_units))
        # ====== initialize states ====== #
        if init is None:
            # reshape to (1, -1) if possible
            if isinstance(self.state_init, np.ndarray):
                if self.state_init.ndim == ndim - 2:
                    shape = (1,) + self.state_init.shape
                    self.state_init = self.state_init.reshape(*shape)
                elif self.state_init.ndim != ndim - 1:
                    raise ValueError('state_init as numpy ndarray has %d dimension '
                                     ', but the given input require %d dimension'
                                     '.' % (self.state_init.ndim, ndim - 1))
            # create init of state
            config.create_params(self.state_init,
                                 shape=(1,) + input_shape[2:-1] + (self.num_units,),
                                 name='init',
                                 nnops=self,
                                 roles=INITIAL_STATE)
        else:
            if K.get_shape(init)[-1] != self.num_units or K.ndim(init) != ndim - 1:
                raise Exception('Initialization of state must has trialing '
                                'dimension = %d, and number of dimension = %d, '
                                'but given "init" has %d and %d.' % (self.num_units,
                                ndim - 1, K.get_shape(init)[-1], K.ndim(init)))
            # turn off repeat_states if batch_size already included
            if K.get_shape(init)[0] != 1:
                self.repeat_states = False
            self.init = init
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
