from __future__ import print_function, division, absolute_import

import types
import inspect
from collections import Mapping

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils import as_tuple, is_number, flatten_list, ctext
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

  def set_nnops(self, ops):
    if isinstance(ops, (tuple, list)): # remove None values
      ops = [o for o in ops if o is not None]
    ops = as_tuple(ops, t=NNOp)
    for o in ops:
      name = o.name.split('/')[-1]
      self.get_variable_nnop(name=name, initializer=o)
    return self

# ===========================================================================
# Implementation
# ===========================================================================
class StochasticDepth(Container):
  pass

class TimeDistributed(Container):
  """ TimeDistributed """

  def __init__(self, ops, time_axis=1, **kwargs):
    super(TimeDistributed, self).__init__(**kwargs)
    self.set_nnops(ops)
    self.time_axis = int(time_axis)

  def _apply(self, *args, **kwargs):
    pass
    K.Scan

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
    all_outputs = []
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
      type_format = '%-' + str(max(len(type(o).__name__) for o in self.nnops)) + 's'
      name_format = '%-' + str(max(len(o.name) for o in self.nnops)) + 's'
    # ====== start apply each NNOp ====== #
    for i, op in enumerate(self.nnops):
      if i == 0:
        x = op(*args, **kwargs)
      else:
        x = op(x, **kwargs)
      all_outputs.append(x)
      # print after finnish the op
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
    return all_outputs if self.return_all_layers else x

  def _transpose(self):
    transpose_ops = []
    for i in self.nnops:
      if hasattr(i, 'T'):
        transpose_ops.append(i.T)
      elif not self.strict_transpose:
        transpose_ops.append(i)
    # reversed the order of ops for transpose
    transpose_ops = list(reversed(transpose_ops))
    seq = Sequence(transpose_ops, debug=self.debug,
                   name=self.name + '_transpose')
    return seq
