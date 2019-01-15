import os
import shutil
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils import (is_string, is_number, ctext, wprint,
                        is_primitives, is_pickleable, FuncDesc)

# ===========================================================================
# Helper
# ===========================================================================
def _check_dtype(dtype):
  if hasattr(dtype, '__call__'):
    return dtype
  # ====== check dtype ====== #
  if dtype is None:
    dtype = K.floatX
  elif isinstance(dtype, np.dtype) or is_string(dtype):
    dtype = str(dtype)
  elif isinstance(dtype, VariableDesc):
    dtype = dtype.dtype
  elif isinstance(dtype, tf.DType):
    dtype = dtype.base_dtype.name
  return dtype

def _check_shape(s):
  if hasattr(s, '__call__'):
    return s
  if is_number(s) or s is None:
    s = (s,)
  elif isinstance(s, np.ndarray):
    s = s.tolist()
  return tuple([int(i) if is_number(i) else None for i in s])

def _shape_compare(shape1, shape2):
  """Return True if shape1 == shape2"""
  if len(shape1) != len(shape2): # different ndim
    return False
  for s1, s2 in zip(shape1, shape2):
    if s1 is None or s2 is None or s1 == -1 or s2 == -1:
      continue
    if s1 != s2:
      return False
  return True

# ===========================================================================
# Main Descriptor
# ===========================================================================
class VariableDesc(object):
  """ VariableDesc
  Store all the necessary information to create placeholder as input
  to any ComputationalGraph.

  Parameters
  ----------
  shape: tuple, list, TensorVariable, call-able
      if TensorVariable is given, shape and dtype will be taken from
      given variable. if a call-able object is given, the object must
      return shape information when called without any argument.
  dtype: str, numpy.dtype, call-able
      dtype of input variable
  name: str, None, call-able
      specific name for the variable

  Note
  ----
  This object is pickle-able and comparable
  """

  def __init__(self, shape, dtype=None, name=None):
    super(VariableDesc, self).__init__()
    # ====== placeholder ====== #
    self.__placeholder = None
    self._name = name if name is None else str(name)
    # Given a TensorVariabe, we don't want to pickle TensorVariable,
    # so copy all necessary information
    if K.is_tensor(shape):
      if dtype is None:
        self._dtype = _check_dtype(shape.dtype)
      self._shape = shape.shape.as_list()
      # store the placeholder so don't have to create it again
      self.__placeholder = shape
    # input the VariableDesc directly
    elif isinstance(shape, VariableDesc):
      self._shape = shape.shape
      self._dtype = shape.dtype if dtype is None \
          else _check_dtype(dtype)
      if shape.__placeholder is not None:
        self.__placeholder = shape.__placeholder
    # input regular information flow
    else:
      self._shape = _check_shape(shape)
      self._dtype = _check_dtype(dtype)

  # ==================== pickle ==================== #
  def __getstate__(self):
    return (self._shape, self._dtype, self._name)

  def __setstate__(self, states):
    (self._shape, self._dtype, self._name) = states
    self.__placeholder = None

  # ==================== properties ==================== #
  def set_placeholder(self, plh):
    if not K.is_placeholder(plh):
      raise ValueError("a placholder must be specified.")
    if plh.shape.as_list() == self.shape and \
    _check_dtype(plh.dtype) == self.dtype:
      self.__placeholder = plh
    else:
      raise ValueError("This VariableDesc require input with shape=%s,"
                       "and dtype=%s, but given a placholder with shape=%s, "
                       "dtype=%s." % (str(self.shape), self.dtype,
                      str(plh.shape.as_list()), _check_dtype(plh.dtype)))
    return self

  @property
  def placeholder(self):
    if self.__placeholder is None:
      self.__placeholder = K.placeholder(
          shape=self.shape, dtype=self.dtype, name=self.name)
    return self.__placeholder

  @property
  def name(self):
    return self._name

  @property
  def shape(self):
    s = self._shape() if hasattr(self._shape, '__call__') \
        else self._shape
    return tuple(s)

  @property
  def dtype(self):
    return self._dtype() if hasattr(self._dtype, '__call__') \
        else self._dtype

  # ==================== override ==================== #
  def __str__(self):
    return "<%s - name:%s shape:%s dtype:%s init:%s>" % \
    (ctext('VarDesc', 'cyan'), ctext(str(self.name), 'yellow'),
        str(self.shape), str(self.dtype),
     False if self.__placeholder is None else True)

  def __repr__(self):
    return self.__str__()

  def is_equal(self, other):
    # ====== compare to a TensorVariable ====== #
    if K.is_tensor(other):
      other = VariableDesc(
          shape=other.shape.as_list(),
          dtype=_check_dtype(other.dtype))
    # ====== compare to a VariableDesc ====== #
    if isinstance(other, VariableDesc):
      if _shape_compare(self.shape, other.shape) \
      and self.dtype == other.dtype:
        return True
    # ====== compare to a shape tuple (ignore the dtype) ====== #
    elif isinstance(other, (tuple, list)):
      return True if _shape_compare(self.shape, other) else False
    return False

class NNOpDesc(object):

  def __init__(self, nnop_type):
    super(NNOpDesc, self).__init__()
    assert isinstance(nnop_type, type)
    self._nnop_type = nnop_type
    self._save_states = {}

    # mapping: name -> VariableDesc, or Primitives
    self._kwargs_desc = OrderedDict()
    # mapping: variable_name -> (tensorflow_name, 'tensor' or 'variable')
    self._variable_info = OrderedDict()

    # this is special tricks, the unpickled ops stay useless
    # until its variables are restored, but if we restore the
    # variable right away, it create a session and prevent
    # any possibility of running tensorflow with multiprocessing
    # => store the _restore_vars_path for later, and restore
    # the variable when the NNOp is actually in used.
    self._set_restore_info(None, False)

  def _set_restore_info(self, vars_path, delete_after):
    self._restore_vars_path = vars_path
    self._delete_vars_folder = bool(delete_after)
    return self

  def _restore_variables(self):
    """ This method can be called anywhere to make sure
    the variable related to this NNOp is restored after
    pickling.
    """
    if hasattr(self, '_restore_vars_path') and \
    self._restore_vars_path is not None:
      folder_path = os.path.dirname(self._restore_vars_path)
      if os.path.exists(folder_path):
        K.restore_variables(self._restore_vars_path)
        # delete cached folder if necessary
        if self._delete_vars_folder:
          shutil.rmtree(folder_path)
      else:
        wprint("NNOp: '%s' cannot restore variables from path: '%s'"
               (self.name, folder_path))
      # reset info
      self._set_restore_info(None, False)

  def __setattr__(self, name, value):
    # this record all assigned attribute to pickle them later
    # check hasattr to prevent recursive loop at the beginning before
    # __init__ is called
    if hasattr(self, '_save_states'):
      if name not in ('_save_states',):
        if is_primitives(value, inc_ndarray=True,
                         exception_types=[NNOpDesc, FuncDesc]) or \
        (hasattr(value, '__call__') and is_pickleable(value)):
          self._save_states[name] = value
    return super(NNOpDesc, self).__setattr__(name, value)
