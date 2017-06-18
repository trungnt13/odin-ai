from __future__ import division, absolute_import

import tensorflow as tf

from odin import backend as K

from .base import NNOp, NNTransposeOps, _nnops_initscope


# ===========================================================================
# Flatten
# ===========================================================================
class Flatten(NNOp):
    """ Flatten the array from the right.
    i.e. turn shape=(128,28,28) with outdim=2 into shape=(128, 784)
    """

    @_nnops_initscope
    def __init__(self, outdim=2, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.outdim = outdim

    def _apply(self, x):
        return K.flatten(x, outdim=self.outdim)

    def _transpose(self):
        return Reshape(shape=self.input_shape_ref)


# ===========================================================================
# REshape
# ===========================================================================
class Reshape(NNOp):

    @_nnops_initscope
    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.shape = shape

    def _apply(self, x):
        return K.reshape(x, shape=self.shape)

    def _transpose(self):
        return Reshape(shape=self.input_shape_ref)


class Dimshuffle(NNOp):

    @_nnops_initscope
    def __init__(self, pattern, **kwargs):
        super(Dimshuffle, self).__init__(**kwargs)
        self.pattern = pattern

    def _apply(self, x):
        return K.dimshuffle(x, pattern=self.pattern)

    def _transpose(self):
        return Reshape(shape=self.input_shape_ref)


class Squeeze(NNOp):

    @_nnops_initscope
    def __init__(self, axis, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis

    def _apply(self, x):
        input_shape = x.get_shape()
        if input_shape[self.axis] != 1:
            raise ValueError('The squeeze axis=%d must be 1, but got %d instead' %
                             (self.axis, input_shape[self.axis]))
        return tf.squeeze(x, axis=self.axis)

    def _transpose(self):
        return InvertSqueeze(self)


class InvertSqueeze(NNTransposeOps):

    def _apply(self, X):
        ndim = len(self.T.input_shape)
        axis = self.T.axis % ndim
        pattern = ['x' if i == axis
                   else (i - 1 if i > axis else i)
                   for i in range(ndim)]
        return K.dimshuffle(X, pattern)
