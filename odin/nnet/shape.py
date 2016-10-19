from __future__ import division, absolute_import


import numpy as np

from odin import backend as K
from odin.utils.decorators import autoinit

from .base import NNOps, NNConfig


class FlattenLeft(NNOps):
    """ Flatten the array from the left.
    i.e. turn shape=(128,28,28) with outdim=2 into shape=(3584, 28)
    """

    @autoinit
    def __init__(self, outdim=2, **kwargs):
        super(FlattenLeft, self).__init__(**kwargs)

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x))
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        other_shape = tuple([input_shape[i]
                             for i in range(K.ndim(x) - self.outdim + 1,
                                            K.ndim(x))])
        return K.reshape(x, (-1,) + other_shape)

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)


class FlattenRight(NNOps):
    """ Flatten the array from the right.
    i.e. turn shape=(128,28,28) with outdim=2 into shape=(128, 784)
    """

    @autoinit
    def __init__(self, outdim=2, **kwargs):
        super(FlattenRight, self).__init__(**kwargs)

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x))
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        other_shape = tuple([input_shape[i]
                             for i in range(self.outdim - 1)])
        n = np.prod(input_shape[(self.outdim - 1):])
        return K.reshape(x, other_shape + (n,))

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)


class Reshape(NNOps):

    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.shape = shape

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x))
        return config

    def _apply(self, x):
        return K.reshape(x, shape_=self.shape)

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)


class Dimshuffle(NNOps):

    def __init__(self, pattern, **kwargs):
        super(Dimshuffle, self).__init__(**kwargs)
        self.pattern = pattern

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x))
        return config

    def _apply(self, x):
        return K.dimshuffle(x, pattern=self.pattern)

    def _transpose(self):
        shape = tuple([-1 if i is None else i for i in self.input_shape])
        return Reshape(shape)


class Squeeze(NNOps):

    def __init__(self, axis, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x))
        return config

    def _apply(self, x):
        return K.squeeze(x, axis=self.axis)

    def _transpose(self):
        ndim = len(self.input_shape)
        axis = self.axis % ndim
        pattern = ['x' if i == axis
                   else (i - 1 if i > axis else i)
                   for i in range(ndim)]
        return Dimshuffle(pattern=pattern)
