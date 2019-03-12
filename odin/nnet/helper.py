from __future__ import print_function, division, absolute_import

import types
import inspect
from collections import Mapping
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.nnet.base import Container
from odin.utils import (as_tuple, is_number, flatten_list, ctext,
                        axis_normalize)

# ===========================================================================
# Implementation
# ===========================================================================
class Residual(Container):
  pass

class StochasticDepth(Container):
  pass

class TimeDistributed(Container):
  """ Time distributed container applying NNOp or list of NNOp
  along given `time_axis` of input tensor

  Parameters
  ----------
  """

  def __init__(self, ops, time_axis=1, backward=False, reverse=False,
               debug=False, **kwargs):
    super(TimeDistributed, self).__init__(**kwargs)
    self.set_nnops(ops)
    self.time_axis = int(time_axis)
    self.backward = bool(backward)
    self.reverse = bool(reverse)
    self.debug = int(debug)

  def _apply(self, X, mask=None):
    def _step_fn(outs, ins):
      return [f(ins) for f in self._apply_ops]
    # ====== need to apply the ops to know initializer information ====== #
    ndim = X.shape.ndims
    axis = axis_normalize(self.time_axis, ndim=ndim)
    with tf.device("/cpu:0"):
      sample = tf.zeros_like(X)
      sample = sample[[slice(None, None) if i != axis else 0
                       for i in range(ndim)]]
    initializer = [tf.zeros_like(f(sample)) for f in self._apply_ops]
    # ====== scan ====== #
    outputs = K.scan_tensors(_step_fn,
                             sequences=X, mask=mask, initializer=initializer,
                             axis=axis,
                             backward=self.backward, reverse=self.reverse,
                             reshape_outputs=True)
    return outputs[0] if len(self._apply_ops) == 1 else outputs

  def _transpose(self):
    return TimeDistributed(ops=[o.T for o in self._apply_ops],
                           time_axis=self.time_axis,
                           backward=self.backward,
                           reverse=self.reverse)

class Parallel(Container):
  """ Parallel

  Parameters
  ----------
  mode : {None, 'concat', 'max', 'min', 'sum', 'mean'}
    None - no post-processing is performed, return the outputs from all NNOp
    'concat' - concatenate the outputs along given `axis`
    'min', 'max' - take the min or max values of the outputs along given `axis`
    'sum' - take sum of the outputs
    'mean' - take mean of the outputs
  """

  def __init__(self, ops, mode='concat', axis=-1,
               debug=False, **kwargs):
    super(Parallel, self).__init__(**kwargs)
    self.set_nnops(ops)
    mode = str(mode).lower()
    assert mode in ('none', 'concat', 'min', 'max', 'sum', 'mean'),\
    "Support `mode` includes: 'none', 'concat', 'min', 'max', 'sum', 'mean';" +\
    " but given: %s" % mode
    self.mode = mode
    self.axis = axis
    self.debug = int(debug)

  def _apply(self, *args, **kwargs):
    with self._debug_mode(*args, **kwargs):
      outputs = []
      for op in self._apply_ops:
        outputs.append(op(*args, **kwargs))
        self._print_op(op)
      # ====== post-processing ====== #
      if self.mode == 'concat':
        ret = tf.concat(outputs, self.axis)
      elif self.mode == 'max':
        ret = outputs[0]
        for o in outputs[1:]:
          ret = tf.maximum(ret, o)
      elif self.mode == 'min':
        ret = outputs[0]
        for o in outputs[1:]:
          ret = tf.minimum(ret, o)
      elif self.mode == 'sum' or self.mode == 'mean':
        outputs[0]
        for o in outputs[1:]:
          ret = ret + o
        if self.mode == 'mean':
          ret = ret / len(outputs)
      else:
        ret = outputs
    return ret

  def _transpose(self):
    pass

class Sequence(Container):

  """ Sequence of Operators

  Parameters
  ----------
  strict_transpose: bool
      if True, only operators with transposed implemented are added
      to tranpose operator
  debug: bool
      if `1`, print NNOp name and its input and output shape
      if `2`, print all information of each NNOp
  return_all_layers: bool
      if True, return the output from all layers instead of only the last
      layer.

  Note
  ----
  You can specify kwargs for a specific NNOp using `params` keywords,
  for example: `params={1: {'noise': 1}}` will applying noise=1 to the
  1st NNOp, or `params={(1, 2, 3): {'noise': 1}}` to specify keywords
  for the 1st, 2nd, 3rd Ops.

  Example
  -------
  """

  def __init__(self, ops, return_all_layers=False,
               strict_transpose=False, debug=False, **kwargs):
    super(Sequence, self).__init__(**kwargs)
    # ====== validate ops list ====== #
    self.set_nnops(ops)
    self.return_all_layers = bool(return_all_layers)
    self.strict_transpose = bool(strict_transpose)
    self.debug = int(debug)

  def _apply(self, *args, **kwargs):
    with self._debug_mode(*args, **kwargs):
      all_outputs = []
      for i, op in enumerate(self._apply_ops):
        if i == 0:
          x = op(*args, **kwargs)
        else:
          x = op(x, **kwargs)
        all_outputs.append(x)
        # print after finnish the op
        self._print_op(op)
    return all_outputs if self.return_all_layers else x

  def _transpose(self):
    transpose_ops = []
    for i in self._apply_ops:
      if hasattr(i, 'T'):
        transpose_ops.append(i.T)
      elif not self.strict_transpose:
        transpose_ops.append(i)
    # reversed the order of ops for transpose
    transpose_ops = list(reversed(transpose_ops))
    seq = Sequence(transpose_ops, debug=self.debug,
                   name=self.name + '_transpose')
    return seq
