from __future__ import division, absolute_import, print_function

import numpy as np

from odin import backend as K
from odin.roles import INITIAL_STATE, WEIGHT, BIAS


from .base import NNConfig, NNOps


class SimpleRecurrent(NNOps):

    def __init__(self, num_units, activation=K.relu,
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0),
                 state_init=K.init.constant(0), **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.num_units = int(num_units)
        self.activation = activation
        self.W_init = W_init
        self.b_init = b_init
        self.state_init = state_init
        self.repeat_states = True

    @K.rnn_decorator(sequences=['X', 'mask'], states=['init'])
    def _apply(self, X, init, mask=None):
        next_states = X + K.dot(init, self.W) + (self.b if hasattr(self, 'b')
                                                 else 0.)
        next_states = self.activation(next_states)
        if mask is not None:
            next_states = K.switch(mask, next_states, init)
        return next_states

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
            config.create_params(self.state_init,
                                 shape=(self.num_units,),
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
        if self.b_init is not None:
            config.create_params(self.b_init,
                                 shape=(self.num_units,),
                                 name='b',
                                 nnops=self,
                                 roles=BIAS)
        return config
