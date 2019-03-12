from __future__ import division, absolute_import

import tensorflow as tf

from odin import backend as K
from odin.nnet.base import NNOp

# ===========================================================================
# Flatten
# ===========================================================================
class Flatten(NNOp):
  """ Flatten the array from the right.
  i.e. turn shape=(128,28,28) with outdim=2 into shape=(128, 784)
  """

  def __init__(self, outdim=2, **kwargs):
    super(Flatten, self).__init__(**kwargs)
    self.outdim = int(outdim)

  def _apply(self, X):
    ndims = X.get_shape().ndims
    if ndims is not None:
      if ndims == self.outdim:
        return X
      elif ndims < self.outdim:
        raise RuntimeError("Input shape: %s, cannot be flatten to %d-D"
          % (str(X.get_shape()), self.outdim))
    return K.flatten(X, outdim=self.outdim)

  def _transpose(self):
    return Reshape(shape=self.input_shape)


# ===========================================================================
# REshape
# ===========================================================================
class Reshape(NNOp):
  """ More flexible version of reshape operation, could be used
  for dimension shuffling as well

  Example
  -------
  x.shape = (25, 08, 12)
  reshape(shape=([1], [2], [0]))
  => x.shape = (08, 12, 25)
  """

  def __init__(self, shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.shape = shape

  def _apply(self, X):
    return K.reshape(X, shape=self.shape)

  def _transpose(self):
    return Reshape(shape=self.input_shape)


class Dimshuffle(NNOp):

  def __init__(self, pattern, **kwargs):
    super(Dimshuffle, self).__init__(**kwargs)
    self.pattern = pattern

  def _apply(self, X):
    return K.dimshuffle(X, pattern=self.pattern)

  def _transpose(self):
    return Reshape(shape=self.input_shape)


class Squeeze(NNOp):

  def __init__(self, axis, **kwargs):
    super(Squeeze, self).__init__(**kwargs)
    self.axis = axis

  def _apply(self, X):
    input_shape = X.shape
    if input_shape[self.axis] != 1:
      raise ValueError('The squeeze axis=%d must be 1, but got %d instead' %
                       (self.axis, input_shape[self.axis]))
    return tf.squeeze(X, axis=self.axis)

  def _transpose(self):
    return InvertSqueeze(self)


# NNTranspose
class InvertSqueeze(NNOp):

  def _apply(self, X):
    ndim = len(self.T.input_shape)
    axis = self.T.axis % ndim
    pattern = ['x' if i == axis
               else (i - 1 if i > axis else i)
               for i in range(ndim)]
    return K.dimshuffle(X, pattern)
