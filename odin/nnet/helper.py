from __future__ import print_function, division, absolute_import

import types
import inspect
from collections import Mapping
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils import (as_tuple, is_number, flatten_list, ctext,
                        axis_normalize)
from odin.utils.decorators import functionable

from .base import NNOp, get_nnop_scope


def _shrink_kwargs(op, kwargs):
  """ Return a subset of kwargs that given op can accept """
  if hasattr(op, '_apply'): #NNOp
    op = op._apply
  elif isinstance(op, functionable): # functionable
    op = op.function
  elif not isinstance(op, types.FunctionType): # call-able object
    op = op.__call__
  sign = inspect.signature(op)
  if any(i.kind == inspect.Parameter.VAR_KEYWORD
         for i in sign.parameters.values()):
    return kwargs
  return {n: kwargs[n] if n in kwargs else p.default
          for n, p in sign.parameters.items()
          if (n in kwargs or p.default != inspect.Parameter.empty) and
             (p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                             inspect.Parameter.VAR_KEYWORD))}

class Residual(NNOp):

  def __init__(self, ops, **kwargs):
    super(Residual, self).__init__(ops, **kwargs)

  def _initialize(self, **kwargs):
    print(self.input_shape, kwargs)
    exit()

  def _apply(self, X, **kwargs):
    pass

# ===========================================================================
# Base class
# ===========================================================================
class Container(NNOp):
  """ Container """

  def __init__(self, **kwargs):
    super(Container, self).__init__(**kwargs)
    self.debug = 0

  def set_nnops(self, ops):
    if isinstance(ops, (tuple, list)): # remove None values
      ops = [o for o in ops if o is not None]
    ops = as_tuple(ops, t=NNOp)
    for o in ops:
      name = o.name.split('/')[-1]
      self.get_variable_nnop(name=name, initializer=o)
    self._apply_ops = ops
    return self

  @contextmanager
  def _debug_mode(self, *args, **kwargs):
    args_desc = [tuple(x.shape.as_list()) if hasattr(x, 'get_shape') else str(x)
                 for x in self._current_args]
    kwargs_desc = {
        k: tuple(v.shape.as_list()) if hasattr(v, 'get_shape') else str(v)
        for k, v in self._current_kwargs.items()}
    # ====== print debug ====== #
    if self.debug > 0:
      print('**************** Start: %s ****************' %
          ctext(self.name, 'cyan'))
      print("First input:", ctext(str(args_desc) + ' ' + str(kwargs_desc), 'yellow'))
    # ====== running ====== #
    self._debug_ops = []
    yield
    # ====== print each op ====== #
    if len(self._debug_ops) > 0:
      type_format = '%-' + str(max(len(type(o).__name__) for o in self._debug_ops)) + 's'
      name_format = '%-' + str(max(len(o.name) for o in self._debug_ops)) + 's'
      for op in self._debug_ops:
        if self.debug == 1:
          print('[' + type_format % op.__class__.__name__ + ']',
                ctext(name_format % op.name, 'cyan'),
                "out:%s" % ctext(op.output_shape, 'yellow'))
        elif self.debug >= 2:
          print(str(op))
    # ====== ending and return ====== #
    if self.debug > 0:
      print('**************** End: %s ****************' %
            ctext(self.name, 'cyan'))

  def _print_op(self, op):
    # print after finish the op at each step
    self._debug_ops.append(op)

# ===========================================================================
# Implementation
# ===========================================================================
class StochasticDepth(Container):
  pass

class TimeDistributed(Container):
  """ Time Distributed container
  """

  def __init__(self, ops, time_axis=1, backward=False, reverse=False, **kwargs):
    super(TimeDistributed, self).__init__(**kwargs)
    self.set_nnops(ops)
    self.time_axis = int(time_axis)
    self.backward = bool(backward)
    self.reverse = bool(reverse)

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
  mode : {None, 'concat', 'max', 'min'}
    None - no post-processing is performed, return the outputs from all NNOp
    'concat' - concatenate the outputs along given `axis`
    'min', 'max' - take the min or max values of the outputs along given `axis`
  """

  def __init__(self, ops, mode='concat', axis=-1,
               debug=False, **kwargs):
    super(Parallel, self).__init__(**kwargs)
    self.set_nnops(ops)
    mode = str(mode).lower()
    assert mode in ('none', 'concat', 'min', 'max'),\
    "Support `mode` includes: 'none', 'concat', 'min', 'max'; but given: %s" % mode
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
