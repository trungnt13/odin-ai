from __future__ import division, absolute_import


import numpy as np

from odin import backend as K
from odin.utils.decorators import autoinit

from .base import NNOps, NNConfig


class Flatten(NNOps):

    @autoinit
    def __init__(self, outdim=2, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def _initialize(self, input_shape):
        config = NNConfig(input_shape=input_shape)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(input_shape=input_shape)
        other_shape = tuple([input_shape[i]
                             for i in range(K.ndim(x) - self.outdim + 1,
                                            K.ndim(x))])
        return K.reshape(x, (-1,) + other_shape)

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)


class Reshape(NNOps):

    @autoinit
    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)

    def _initialize(self, input_shape):
        config = NNConfig(input_shape=input_shape)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(input_shape=input_shape)
        return K.reshape(x, shape_=self.shape)

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)


class Dimshuffle(NNOps):

    @autoinit
    def __init__(self, pattern, **kwargs):
        super(Dimshuffle, self).__init__(**kwargs)

    def _initialize(self, input_shape):
        config = NNConfig(input_shape=input_shape)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(input_shape=input_shape)
        return K.dimshuffle(x, pattern=self.pattern)

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)
