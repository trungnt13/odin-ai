from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils import is_callable

from .base import NNOp


_SUPPORT_REDUCE_FUNCTION = [
    tf.reduce_sum,
    tf.reduce_mean,
    tf.reduce_max,
    tf.reduce_min,
    tf.reduce_prod,
    tf.reduce_logsumexp,
    tf.reduce_all,
    tf.reduce_any
]

class Reduce(NNOp):
  """ support all Reduce function in tensorflow"""

  def __init__(self, fn, axis=None, keep_dims=False, **kwargs):
    super(Reduce, self).__init__(**kwargs)
    if not is_callable(fn):
      raise ValueError("`fn` must be callable")
    if fn not in _SUPPORT_REDUCE_FUNCTION:
      raise ValueError("Not support for function type: %s, all support function include: %s" %
        (str(fn), ', '.join([i.__name__ for i in _SUPPORT_REDUCE_FUNCTION])))
    self.fn = fn
    self.axis = axis
    self.keep_dims = bool(keep_dims)

  def _transpose(self):
    raise NotImplementedError

  def _apply(self, X):
    return self.fn(X, axis=self.axis, keep_dims=self.keep_dims)

class Repeat(NNOp):
  """ Repeat a dimension of a tensor

  If x has shape (s1, s2, s3) and axes=(1, -1), the output
  will have shape (s1, s2 * n[0], s3 * n[1]).

  Parameters
  ----------
  n : {int, list of int}
    each number of repeatation according to the axes
  axes : {int, list or int}
    all axes for repeating
  """

  def __init__(self, n, axes):
    super(Repeat, self).__init__()
    self.n = n
    self.axes = axes

  def _transpose(self):
    raise NotImplementedError

  def _apply(self, X):
    return K.repeat(X, n=self.n, axes=self.axes)
