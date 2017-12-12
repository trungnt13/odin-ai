from __future__ import division, absolute_import, print_function

import os
import re
import sys
import shutil
import inspect
import numbers
import warnings
from itertools import chain
from functools import wraps
from collections import OrderedDict, defaultdict, Mapping
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from six.moves import zip, range, cPickle
from six import add_metaclass, types, string_types

import numpy as np

from odin import backend as K
from odin.utils.decorators import functionable
from odin.utils import (as_tuple, as_list, uuid, cache_memory, is_number,
                        is_string, is_path, is_primitives, ctext,
                        flatten_list, get_all_files, is_pickleable,
                        FuncDesc, dummy_formatter, type_path,
                        get_module_from_path)
from odin.backend.role import (add_role, has_roles, Parameter, Weight, Bias)

import tensorflow as tf


# ===========================================================================
# Global NNOp manager
# ===========================================================================
def get_all_nnops(scope=None, op_type=None):
  """ Return a dictionary of (name, nnops) for all created NNOp """
  allops = list(NNOp._ALL_NNOPS.values())
  if scope is not None:
    if not is_string(scope):
      scope = scope.name
    allops = [o for o in allops if o.name[:len(scope)] == scope]
  if op_type is not None:
    op_type = [i for i in as_tuple(op_type)
               if is_string(op_type) or issubclass(op_type, NNOp)]
    allops = [o for o in allops
              if any(o.__class__.__name__ == t if is_string(t)
                     else isinstance(o, t)
                     for t in op_type)]
  return allops


def _assign_new_nnop(nnop):
  if not isinstance(nnop, NNOp):
    raise ValueError("The new assigned NNOp must be instance of odin.nnet.NNOp "
                     ", but the given object has type: %s" %
                     str(type(nnop)))
  name = nnop.name
  if not nnop.is_initialized:
    raise RuntimeError("Given NNOp with name: '%s' has not been initialized"
                       % name)
  if name in NNOp._ALL_NNOPS:
    raise RuntimeError("Another NNOp of type: '%s', and name: '%s' has "
                       "already existed." %
                       (type(NNOp._ALL_NNOPS[name]), name))
  NNOp._ALL_NNOPS[name] = nnop


# ===========================================================================
# Context manager
# ===========================================================================
__ARGS_SCOPE_STACK = [defaultdict(dict)]
# each element is a list [scope_name, prefix_name, current_id]
_NNOP_SCOPE_STACK = []


def get_nnop_scope():
  if len(_NNOP_SCOPE_STACK) == 0:
    return ['', '', uuid()]
  return _NNOP_SCOPE_STACK[-1]


def get_args_scope():
  return __ARGS_SCOPE_STACK[-1].copy()


@contextmanager
def args_scope(ops, **kwargs):
  """Stores the default arguments for the given set of applied_nnops.

  For usage, please see examples at top of the file.

  Parameters
  ----------
  applied_nnops: List or tuple string, type, or NNOp
      a dictionary containing the current scope. When list_ops_or_scope is a
      dict, kwargs must be empty. When list_ops_or_scope is a list or tuple,
      then every op in it need to be decorated with @add_arg_scope to work.
  **kwargs: keyword=value that will define the defaults for each op in
      list_ops. All the ops need to accept the given set of arguments.

  Return
  ------
  the current_scope, which is a dictionary of {op: {arg: value}}

  Raises
  ------
  TypeError: if list_ops is not a list or a tuple.
  ValueError: if any op in list_ops has not be decorated with @add_arg_scope.

  Note
  ----
  if the name scope is given, an increasement ID is generated for
  duplicated NNOp instead of UUID.
  """
  # ====== update Arguments Scope ====== #
  ops = as_tuple(ops)
  # copy prevous scopes
  args_scope = defaultdict(dict)
  for i, j in __ARGS_SCOPE_STACK[-1].items():
    args_scope[i] = j.copy()
  # update new scopes
  for o in ops:
    args_scope[o].update(kwargs)
  __ARGS_SCOPE_STACK.append(args_scope)
  # ====== return the scope ====== #
  yield None
  # ====== reset everything ====== #
  __ARGS_SCOPE_STACK.pop()


@contextmanager
def nnop_scope(scope, prefix='', reuse=None):
  """
  Parameters
  ----------
  scope: string
      the name of current scope, new NNOp will be created as "scope/name"
  prefix: string
      prefix for NNOp name, just in case a name is not given when NNOp is
      initialized
  reuse: bool
      whether reuse variables for tensorflow.variable_scope

  Example
  -------
  >>> X = K.variable(x, name='x')
  >>> with N.nnop_scope(scope='s1'):
  ...     f1 = N.Dense(8)
  ...     f1(X)
  ...     with N.nnop_scope(scope='s2'):
  ...         f2 = N.Dense(25)
  ...         f2(X)
  ...         with N.nnop_scope(scope='s1', prefix='S'):
  ...             f3 = N.Dense(12)
  ...             f3(X)
  >>> with N.nnop_scope(scope='s1/s2', prefix='S'):
  ...     f4 = N.Dense(num_units=13)
  ...     f4(X)
  >>> # f1: s1/Dense_0
  >>> # f2: s1/s2/Dense_0
  >>> # f3: s1/s2/Dense_S0
  >>> # f4 == f3
  >>> print(N.get_all_nnops(scope='s1/s2')) # contains 2 NNOp
  >>> print(N.get_all_nnops(scope='s1')) # contains 3 NNOp
  """
  if not is_string(scope) or len(scope) == 0:
    raise ValueError("`scope` must be string type, length > 0.")
  if not is_string(prefix):
    raise ValueError("`prefix` must be string type.")
  # ====== prepare Name Scope ====== #object
  curr_scope, curr_prefix, curr_opID = get_nnop_scope()
  # NO duplicate scope
  if scope not in curr_scope:
    curr_scope = scope if len(curr_scope) == 0 else \
        curr_scope + '/' + scope
  # name scope
  _NNOP_SCOPE_STACK.append([curr_scope, prefix, 0])
  # ====== return the scope ====== #
  var_scope = tf.get_variable_scope().name
  # NO repeating the scope in variable scope
  if any(s == scope for s in var_scope.split('/')): # this may cause error
    yield curr_scope
  else:
    with tf.variable_scope(scope, reuse=reuse):
      yield curr_scope
  # ====== reset everything ====== #
  _NNOP_SCOPE_STACK.pop()


# ===========================================================================
# Helper
# ===========================================================================
def _check_shape(s):
  if hasattr(s, '__call__'):
    return functionable(s)
  if is_number(s) or s is None:
    s = (s,)
  elif isinstance(s, np.ndarray):
    s = s.tolist()
  return tuple([int(i) if is_number(i) else None for i in s])


def _check_dtype(dtype):
  if hasattr(dtype, '__call__'):
    return functionable(dtype)
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


def _shape_compare(shape1, shape2):
  """Return True if shape1 == shape2"""
  if len(shape1) != len(shape2):
    return False
  for s1, s2 in zip(shape1, shape2):
    if s1 != s2:
      return False
  return True


# ===========================================================================
# Input descriptor
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
      self._shape = shape.get_shape().as_list()
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
    if plh.get_shape().as_list() == self.shape and \
    _check_dtype(plh.dtype) == self.dtype:
      self.__placeholder = plh
    else:
      raise ValueError("This VariableDesc require input with shape=%s,"
                       "and dtype=%s, but given a placholder with shape=%s, "
                       "dtype=%s." % (str(self.shape), self.dtype,
                      str(plh.get_shape().as_list()), _check_dtype(plh.dtype)))
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
          shape=other.get_shape().as_list(),
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


# ===========================================================================
# Main Ops
# ===========================================================================
class _NNOp_Meta(ABCMeta):
  """ arguments scope for the NNOp

  Note
  ----
  you can only modify the arguments and kwarguments using __call__
  from MetaClass, not __new__ of instance class.
  """
  def __new__(mcs, name, bases, class_dict):
    private = {'T', 'apply', '__call__', '__getstate__', '__setstate__',
               '__getnewargs__', 'get', 'get_variable', '__setattr__',
               'input_shape', 'placeholders', 'last_output'}
    if name != 'NNOp':
      for attr in private:
        if attr in class_dict:
          raise RuntimeError("[Class:%s]The behaviour of NNOp is "
              "restricted to ensure properly operations, the following "
              "methods or properties cannot be overrided: '%s'" %
              (ctext(name, 'red'), ctext(attr, 'yellow')))
    return super().__new__(mcs, name, bases, class_dict)

  def __call__(clazz, *args, **kwargs):
    NO_ARGUMENT = '[__no_argument__]'
    # getting the default arguments to check user intentionally override
    # default argument.
    spec = inspect.getargspec(clazz.__init__)
    # ignore the self argument
    default_args = OrderedDict([(i, NO_ARGUMENT) for i in spec.args[1:]])
    if spec.defaults is not None:
      for name, value in zip(spec.args[::-1], spec.defaults[::-1]):
        default_args[name] = value
    # ====== upate the current argument scope ====== #
    # get current scope
    key_name = [clazz, str(clazz), clazz.__name__]
    current_scope = get_args_scope()
    for n in key_name:
      if n in current_scope:
        default_args.update(current_scope[n])
    # ====== update user specified args and kwargs ====== #
    # update the new arguments into default arguments
    new_kwargs = OrderedDict([
        (name, args[i]) if i < len(args) else (name, default)
        for i, (name, default) in enumerate(default_args.items())])
    new_kwargs.update(kwargs)
    # ====== create new instance and __init__ if necessary ====== #
    op = clazz.__new__(clazz, *[], **new_kwargs)
    if not hasattr(op, '_name'):
      raise ValueError("NNOp must be given a name when initialized.")
    # check if op already initialized
    if op.name not in NNOp._ALL_NNOPS:
      clazz.__init__(op, *[], **new_kwargs)
    return op


@add_metaclass(_NNOp_Meta)
class NNOp(object):
  """ Basics of all Neural Network operators

  Properties
  ----------
  name: str
      identity of the operator, this name is the scope for its operator
      and should be unique.
  T: NNOp
      transpose operator of this one (NOTE: some ops does not support
      transpose and raise NotImplementedError)
  parameters: list of variables
      list of all parameters associated with this operator scope

  Abstract
  --------
  _apply(self, X, **kwargs): resulted variables
      apply take a list of variables and custom parameters to compute
      output variables
  _initialize(self, **kwargs):
      create parameters

  Override
  --------
  _transpose(self): NNOp
      return another NNOp which is transposed version of this ops

  Note
  ----
  All NNOp are pickle-able!
  You must use: protocol=cPickle.HIGHEST_PROTOCOL when dump NNOp.
  if NNOp is applied to a list of inputs, it will process each input seperated.
  """
  _ALL_NNOPS = {}

  def __new__(clazz, *args, **kwargs):
    # ====== cPickle call __new__ ====== #
    if len(args) == 1 and len(kwargs) == 0 and \
    (is_string(args[0]) and '[__name__]' in args[0]):
      name = args[0].replace('[__name__]', '')
      # Found predefined NNOP
      if name in NNOp._ALL_NNOPS:
        instance = NNOp._ALL_NNOPS[name]
        if not isinstance(instance, clazz):
          raise RuntimeError("Found duplicated NNOp with type: '%s', "
              "which is different from pickled type: '%s'" %
              (type(instance), clazz))
        return instance
      # just create new instance
      return super(NNOp, clazz).__new__(clazz)
    # New for first time create instance
    # ====== update Op name if it is None ====== #
    name = kwargs.get('name', None)
    op_scope = get_nnop_scope()
    # automatic generate name
    if name is None:
      name = clazz.__name__ + '_' + str(op_scope[1]) + str(op_scope[2])
    # regulation for the NNOp name
    elif is_string(name):
      if '/' in name or ':' in name:
        raise ValueError("NNOp cannot contain '\\' or ':', given name is: %s" % name)
    else:
      raise ValueError("name for NNOp must be string, but given name "
                       "has type: %s" % name)
    # ====== add scope to name ====== #
    if len(op_scope[0]) > 0:
      name = op_scope[0] + '/' + name
      op_scope[-1] += 1
    # ====== check duplicated Op name ====== #
    if name in NNOp._ALL_NNOPS:
      old_clazz = NNOp._ALL_NNOPS[name].__class__
      if clazz != old_clazz:
        raise RuntimeError("Found predefined NNOp with type: %s, but "
            "the new NNOp has type: %s" % (old_clazz, clazz))
      return NNOp._ALL_NNOPS[name]
    # ====== allocate new Op ====== #
    new_op = super(NNOp, clazz).__new__(clazz)
    new_op._name = name
    # this store spontanious args and kwargs feeded to apply()
    new_op._current_args = ()
    new_op._current_kwargs = {}
    new_op._device = None
    # all save-able attributes of NNOp store here
    new_op._save_states = {'_name': name}
    return new_op

  def __init__(self, **kwargs):
    # list of positional VariableDesc, or Primitives
    self._args_desc = []
    # mapping: name -> VariableDesc, or Primitives
    self._kwargs_desc = OrderedDict()
    # mapping: ','.join(id(tensor)) -> output
    self._cache_outputs = {}
    self._last_input_footprint = ''
    self._transpose_ops = None
    self._is_initialized = False
    # this store the sorted, concatenated name of
    # all initialized variables belong to this NNOp
    # if there is change in list of Variables, the
    # the NNOp will automatically initialize all the
    # variable again.
    self._is_initialized_all_variables = ''
    # mapping: variable_name -> (tensorflow_name, 'tensor' or 'variable')
    self._variable_info = OrderedDict()
    # special flags to detect if cPickle called with protocol >= 2
    self._new_args_called = False
    # this is special tricks, the unpickled ops stay useless
    # until its variables are restored, but if we restore the
    # variable right away, it create a session and prevent
    # any possibility of running tensorflow with multiprocessing
    # => store the _restore_vars_path for later, and restore
    # the variable when the NNOp is actually in used.
    self._restore_vars_path = None
    self._delete_vars_folder = False

  # ==================== pickling method ==================== #
  def __getstate__(self):
    if not self._new_args_called:
      raise RuntimeError(
          "You must use argument `protocol=cPickle.HIGHEST_PROTOCOL` "
          "when using `pickle` or `cPickle` to be able pickling NNOp.")
    self._new_args_called = False
    # add nnops here so all related NNOps are saved
    return self._save_states, self.nnops

  def __setstate__(self, states):
    # ====== default attribute ====== #
    self._current_args = ()
    self._current_kwargs = {}
    self._cache_outputs = {}
    # ====== save states ====== #
    self._save_states, nnops = states
    for key, val in self._save_states.items():
      setattr(self, key, val)
    # ====== check exist NNOp ====== #
    if self.name not in NNOp._ALL_NNOPS:
      _assign_new_nnop(self)
    elif NNOp._ALL_NNOPS[self.name] != self:
      raise RuntimeError("Mismatch NNOp, two NNOps with the same name "
                         "are initizlied:\n%s\nis different from:\n%s" %
                         (str(NNOp._ALL_NNOPS[self.name]), str(self)))

  def __getnewargs__(self):
    self._new_args_called = True
    return ('[__name__]' + self.name,)

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
        # delte cached folder if necessary
        if self._delete_vars_folder:
          shutil.rmtree(folder_path)
        self._restore_vars_path = None
        self._delete_vars_folder = False

  # ==================== properties ==================== #
  def get(self, name, nnop=False):
    """"Simple shortcut for getting defined Variable, or NNOp
    within the scope of this `NNOp`

    Parameters
    ----------
    name : string
        the name, or part of the name of the `Variable` or `NNOp`
    nnop : bool
        if you want to get a `NNOp` with given name instead of
        `Variable`
    """
    # ====== get variable ====== #
    if not nnop:
      if isinstance(name, bytes):
        name = str(name, 'utf-8')
      elif not is_string(name):
        raise ValueError("`name` must be string.")
      if name not in self._variable_info:
        raise ValueError("Variable with name: '%s' hasn't been created." % name)
      return self.get_variable(name)
    # ====== nnop ====== #
    for op in self.nnops:
      if name == op.name:
        return op
    raise ValueError("Cannot find `NNOp` with name: '%s'" % name)

  @cache_memory
  def get_variable(self, name, shape=None, initializer=None, roles=[]):
    """
    Parameters
    ----------
    name: str
        name for the variable
    shape: tuple, list
        expected shape for given variable
    initializer: variable, numpy.ndarray, function
        specification for initializing the weights, if a function
        is given, the arguments must contain `shape`
    roles: `odin.backend.role.Role`
        categories of this variable
    """
    self._restore_variables() # restore variable first
    if name in self.__dict__:
      raise RuntimeError("name='%s' has been defined in dictionary of "
                         "NNOp: '%s', type: %s" %
                         (name, self.name, self.__class__.__name__))
    if shape is not None:
      # convert to tuple if needed
      shape = as_tuple(shape)
      if any(d <= 0 or d is None for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") %
            (shape, name))
    #####################################
    # 1. looking for Defined variable.
    if initializer is None and shape is None:
      if name not in self._variable_info:
        raise ValueError("Cannot find variable with name: %s for NNOps "
            "with name: %s" % (name, self.name))
      var_name, t = self._variable_info[name]
      # get variable
      if t == 'variable':
        var = K.get_all_variables(full_name=var_name)
        if len(var) == 0:
          raise RuntimeError("Cannot find variable with name: %s" % var_name)
        var = var[0]
      # get tensor
      elif t == 'tensor':
        name, footprint = var_name
        op = K.get_all_operations(footprint=footprint)
        if len(op) == 0:
          raise RuntimeError("Cannot find any Op with given footprint: %s" % footprint)
        var = op[0]._outputs[int(name.split(':')[-1])]
      # get nnops, use current args and kwargs for initialization
      elif t == 'nnop':
        var = var_name(*self._current_args, **self._current_kwargs)
      # only care about the first variable
      return add_role(var, roles)
    #####################################
    # 2. initializing function.
    if is_string(initializer):
      var = K.get_all_variables(name=initializer)
      if len(var) == 0:
        var = K.get_all_tensors(name=initializer)
      if len(var) == 0:
        raise ValueError("Cannot find any variable or tensor with name: "
            "'%s' for the initializer." % initializer)
      var = var[0]
    elif isinstance(initializer, NNOp):
      var = initializer
    elif hasattr(initializer, '__call__'):
      var = initializer(shape)
    # is a scalar
    elif is_number(initializer):
      var = np.full(shape=shape, fill_value=initializer, dtype='float32')
    # else actual tensor
    else:
      var = initializer
    #####################################
    # 3. Numpy ndarray.
    if isinstance(var, np.ndarray):
      var = K.variable(var, shape=shape, name=name)
      self._variable_info[name] = (var.name, 'variable')
    #####################################
    # 4. Shared variable, just check the shape.
    elif K.is_variable(var):
      _shape = var.get_shape().as_list()
      if shape is not None and tuple(shape) != tuple(_shape):
        raise Exception('Require variable with shape=%s, but was given different '
                        'shape=%s, name:%s.' %
                        (str(shape), str(_shape), str(name)))
      self._variable_info[name] = (var.name, 'variable')
    #####################################
    # 5. expression, we can only check number of dimension.
    elif K.is_tensor(var):
      # We cannot check the shape here, Tensor (even shared
      # variables) do not have a fixed compile-time shape. We can check the
      # dimensionality though.
      # Note that we cannot assign a name here. We could assign to the
      # `name` attribute of the variable, but the user may have already
      # named the variable and we don't want to override this.
      if shape is not None and var.get_shape().ndims != len(shape):
        raise Exception("parameter with name=%s has %d dimensions, should be "
                        "%d" % (name, var.get_shape().ndims, len(shape)))
      self._variable_info[name] = ((var.name, K.get_operation_footprint(var.op)),
                                   'tensor')
    elif isinstance(var, NNOp):
      self._variable_info[name] = (var, 'nnop')
    #####################################
    # 6. Exception.
    else:
      raise RuntimeError("cannot initialize parameters: 'spec' is not "
                         "a numpy array, a Tensor expression, a call-able "
                         ", or variable name as string (given type: %s)" %
                         type(initializer).__name__)
    # ====== assign annotations ====== #
    if K.is_tensor(var):
      return add_role(var, roles)
    elif isinstance(var, NNOp):
      return var
    else:
      raise ValueError("Unsupport for variable type: %s" %
          type(var).__name__)

  @property
  def name(self):
    return self._name

  @property
  def T(self):
    """ Return new ops which is transpose of this ops """
    if not self.is_initialized:
      raise RuntimeError("NNOp with name:'%s' has not been initialized, "
          "call the Op on any input to first initialize input information."
          % self.name)
    if self._transpose_ops is None:
      self._transpose_ops = self._transpose()
      if not isinstance(self._transpose_ops, NNOp):
        raise ValueError("The _transposed method must return NNOp."
                         "but the returned object has type=%s" %
                         str(type(self._transpose_ops)))
      # hard-fix the name of transpose NNOp
      self._transpose_ops._name = self.name + '_T'
      # this is very brain twisted
      self._transpose_ops._transpose_ops = self
    return self._transpose_ops

  def initialize_all_variables(self):
    """Manually initialize all variables (basically, all Variables
    within this NNOp scope are initialized when you call
    `NNOp.variables`.

    This is just another reasonable shortcut.

    Return
    ------
    All initialized Variables
    """
    return self.variables

  @property
  def variables(self):
    """ Get all variables related to this Op, which include:
     - Initialized Variables
     - Variables belong to related NNOp within this Op.
     - Variables belone to related Tensor within this Op.
     - Variables within the scope of this NNOp.
    """
    self._restore_variables()  # restore variable first
    global_vars = {v.name: v for v in K.get_all_variables()}
    all_vars = []
    tensors = []
    for alias, (name, vtype) in self._variable_info.items():
      if vtype == 'variable':
        all_vars.append(global_vars[name])
      elif vtype == 'tensor':
        tensors.append(self.get_variable(alias))
      elif vtype == 'nnop':
        all_vars += name.variables
    # all variables from tensor
    all_vars += K.ComputationGraph(tensors).variables
    # all variables from NNOp
    for op in self.nnops:
      all_vars += op.variables
    # all variables within the scope
    all_vars += K.get_all_variables(scope=self.name)
    all_vars = tuple(sorted(set(all_vars), key=lambda x: x.name))
    # make sure all variables are initialized
    vars_footprint = ';'.join([v.name for v in all_vars])
    if vars_footprint != self._is_initialized_all_variables:
      self._is_initialized_all_variables = vars_footprint
      K.initialize_all_variables(all_vars)
    # return
    return all_vars

  @property
  def nb_variables(self):
    n = 0
    for p in self.variables:
      n += np.prod(p.get_shape().as_list()).astype('int32')
    return n

  @property
  def parameters(self):
    """ return all TensorVariables which have the PARAMETER role"""
    return [i for i in self.variables if has_roles(i, Parameter)]

  @property
  def nb_parameters(self):
    n = 0
    for p in self.parameters:
      n += np.prod(p.get_shape().as_list()).astype('int32')
    return n

  @property
  def is_initialized(self):
    return self._is_initialized

  @property
  def nnops(self):
    """ Return all NNOp belong to the initialization of this Op
    or within the scope of this Op.
    """
    ops = []
    for name, (var, vtype) in self._variable_info.items():
      if vtype == 'nnop':
        ops.append(var)
    ops += get_all_nnops(scope=self.name)
    # make sure all the nested NNOp has the same _restore_vars_path
    # TODO: this is not good idea, if nested NNOp also restore the
    # variables, it will restore variables multiple times.
    # if hasattr(self, '_restore_vars_path') and \
    # self._restore_vars_path is not None:
    #     for o in ops:
    #         if not hasattr(o, '_restore_vars_path') or \
    #         o._restore_vars_path is None:
    #             o._restore_vars_path = self._restore_vars_path
    #             o._delete_vars_folder = self._delete_vars_folder
    return sorted([o for o in ops if o is not self],
                  key=lambda x: x.name)

  @property
  def placeholders(self):
    """ Create list of placeholder to represent inputs of this NNOp
    """
    x = [i.placeholder for i in self._args_desc
         if isinstance(i, VariableDesc)]
    x += [i.placeholder for i in self._kwargs_desc.values()
         if isinstance(i, VariableDesc)]
    return x[0] if len(x) == 1 else x

  def set_placeholder(self, name, plh):
    if is_number(name):
      return self._args_desc[name].set_placeholder(plh)
    return self._kwargs_desc[name].set_placeholder(plh)

  @property
  def last_output(self):
    if self._last_input_footprint not in self._cache_outputs:
      raise RuntimeError("This NNOp has not been called, and contains "
          "no information about outputs.")
    return self._cache_outputs[self._last_input_footprint]

  @property
  def input_shape(self):
    """NOTE: this input shape is only inferred from last inputs to
    this NNOp,
    since the input argument can has default argument, this input
    shape can change after everytime you call the NNOp"""
    x = [tuple(i.get_shape().as_list())
         for i in as_tuple(self.placeholders)]
    return x[0] if len(x) == 1 else x

  @property
  def output_shape(self):
    """NOTE: this input shape is only inferred from last inputs to
    this NNOp,
    since the input argument can has default argument, this input
    shape can change after everytime you call the NNOp"""
    output = self.last_output
    extract_shape = lambda x: tuple(x.get_shape().as_list()) \
        if hasattr(x, 'get_shape') else \
        (x.shape if hasattr(x, 'shape') else ())
    if isinstance(output, (tuple, list)):
      return [extract_shape(o) for o in output]
    elif isinstance(output, Mapping):
      return OrderedDict([(name, extract_shape(o))
          for name, o in output.items()])
    return extract_shape(output)

  def __setattr__(self, name, value):
    # this record all assigned attribute to pickle them later
    # check hasattr to prevent recursive loop at the beginning before
    # __init__ is called
    if hasattr(self, '_save_states'):
      if name not in ('_save_states', '_cache_outputs',
                      '_current_args', '_current_kwargs',
                      '_last_input_footprint'):
        if is_primitives(value, inc_ndarray=True,
                         exception_types=[NNOp, FuncDesc]) or \
        (hasattr(value, '__call__') and is_pickleable(value)):
          self._save_states[name] = value
    return super(NNOp, self).__setattr__(name, value)

  # ==================== abstract method ==================== #
  def _initialize(self):
    """ This function is only called once, for the first time you
    apply this Ops
    """
    return None

  @abstractmethod
  def _apply(self, X, **kwargs):
    raise NotImplementedError

  def _transpose(self):
    raise NotImplementedError

  # ==================== interaction method ==================== #
  def _check_input_arg(self, x, name):
    """Validate input variable
    Return
    ------
    tuple of (VariableDesc, raw_data)
    VariableDesc: can be None if only given a primitive data types
    raw_data: can be None if only give VariableDesc
    """
    desc = None
    data = None
    # ====== if given tensor, use the provided tensor ====== #
    if K.is_tensor(x) or isinstance(x, VariableDesc):
      desc = x if isinstance(x, VariableDesc) else\
          VariableDesc(shape=x, name=x.name.split(':')[0])
      # positional argument
      if is_number(name):
        if len(self._args_desc) <= name:
          self._args_desc.append(desc)
        curr_desc = self._args_desc[int(name)]
      # keywords argument
      else:
        if name not in self._kwargs_desc:
          self._kwargs_desc[name] = desc
        curr_desc = self._kwargs_desc[name]
      # validating
      if not curr_desc.is_equal(desc):
        raise ValueError("Found variable with description: '%s', given "
            "variable with description: '%s'" %
            (str(curr_desc), str(desc)))
    # ====== if given data, use saved tensor with new data ====== #
    elif isinstance(x, np.ndarray):
      # positional
      if is_number(name):
        if len(self._args_desc) <= name:
          self._args_desc.append(VariableDesc(
              shape=x.shape, dtype=x.dtype,
              name='Inp%d' % name))
        desc = self._args_desc[int(name)]
      # keywords
      else:
        if name not in self._kwargs_desc:
          self._kwargs_desc[name] = VariableDesc(
              shape=x.shape, dtype=x.dtype, name=name)
        desc = self._kwargs_desc[name]
      # validating
      if desc.shape[1:] != x.shape[1:] or \
      np.dtype(desc.dtype) != np.dtype(x.dtype):
        raise ValueError("NNOp has input description: '%s', given "
                         "ndarray: shape=%s dtype=%s" %
                         (str(desc), x.shape, x.dtype))
      # set data
      data = x
    # ====== primitive, keep it simple ====== #
    elif is_primitives(x, inc_ndarray=False):
      # positional
      if is_number(name):
        if len(self._args_desc) <= name:
          self._args_desc.append(x)
        else:
          self._args_desc[name] = x
      # keywords
      else:
        self._kwargs_desc[name] = x
      desc = x
    # ====== Uknown input, ERROR ====== #
    else:
      raise ValueError("The input argument for Model can be: "
          "`Tensor`, `odin.nnet.VariableDesc`, and primitive types"
          " (string, number, boolean, None, numpy.ndarray, numpy.generic)."
          " But the given type is: %s" % type(x))
    return (desc, data)

  def apply(self, *args, **kwargs):
    # self.name can contain Model varable scope, hence,
    # remove the scope here
    self._restore_variables()  # restore variable first
    op_name = self.name.split('/')[-1]
    with nnop_scope(scope=op_name, prefix='', reuse=self.is_initialized):
      # ====== special case, Op name mis match variable scope ====== #
      if not self._is_initialized and \
      self.name != tf.get_variable_scope().name:
        # check Op name match variable scope
        self._name = tf.get_variable_scope().name
      # ====== processing argument information ====== #
      spec = inspect.getargspec(self._apply)
      # adding kwargs_new in Order
      kwargs_new = OrderedDict()
      # if except name outside of spec.args
      extra_name = [] if spec.keywords is None else \
          [name for name in kwargs.keys() if name not in spec.args]
      # get all positional arguments
      for idx, name in enumerate(spec.args[1:] + extra_name):
        if idx < len(args):
          x = args[idx]
        elif name in kwargs:
          x = kwargs[name]
        else:
          continue
        # validate and record name to self._kwargs_desc
        kwargs_new[name] = self._check_input_arg(x=x, name=name)
      # kwargs now contains: arg_name -> (VariableDesc, ndarray(or None))
      kwargs = kwargs_new
      # check if _apply have vargs or keywords
      if spec.varargs is None:
        args = ()
      else:
        args = [self._check_input_arg(a, name=i)
                for i, a in enumerate(args[len(spec.args[1:]):])]
        # add missing positional, using self._args_desc as substitute
        for i, var in enumerate(self._args_desc):
          if i >= len(args):
            args.append((var, None))
      # add missing slot from _kwargs_desc
      for name, var in self._kwargs_desc.items():
        if name not in kwargs:
          kwargs[name] = (var, None)
      # ====== get op inputs and data ====== #
      op_args = []
      op_kwargs = OrderedDict()
      op_data = {}
      footprint = ''
      # footprint created by concat argument name and its
      # object python ID (primitive arugment using str)
      # positional argument
      for i, (desc, dat) in enumerate(args):
        footprint += 'Inp%d:' % i
        if isinstance(desc, VariableDesc):
          desc = desc.placeholder
          footprint += type_path(desc) + '_' + str(id(desc))
          if dat is not None:
            op_data[desc] = dat
        else:
          footprint += type_path(desc) + '_'
          footprint += str(desc)
        footprint += '|'
        op_args.append(desc)
      # kwargs
      for name, (desc, dat) in sorted(kwargs.items(),
                                      key=lambda x: x[0]):
        footprint += name + ':'
        if isinstance(desc, VariableDesc):
          desc = desc.placeholder
          footprint += type_path(desc) + '_' + str(id(desc))
          if dat is not None:
            op_data[desc] = dat
        else: # primitive types
          footprint += type_path(desc) + '_' + str(desc)
        footprint += '|'
        op_kwargs[name] = desc
      # store current arguments
      self._current_args = op_args
      self._current_kwargs = op_kwargs
      # ====== initialize first ====== #
      if not self._is_initialized:
        # call NNOp initialize
        self._initialize()
        self._is_initialized = True
        # only assign new NNOp if it is initialized
        _assign_new_nnop(self)
      # ====== calculate and return outputs ====== #
      # Recall cached output
      if footprint in self._cache_outputs:
        y = self._cache_outputs[footprint]
      # First time generate output given footprint
      else:
        y = self._apply(*op_args, **op_kwargs)
        # record cahced return
        self._cache_outputs[footprint] = y
      # check if op_data given, then evaluate to get the results.
      if len(op_data) > 0:
        # only need to make sure all variables
        # initialized if we need to evaluate some
        # expressions.
        self.initialize_all_variables()
        y = K.eval(y, feed_dict=op_data)
    # ====== reset the current information ====== #
    self._current_args = ()
    self._current_kwargs = {}
    self._last_input_footprint = footprint
    return y

  def __call__(self, *args, **kwargs):
    return self.apply(*args, **kwargs)

  def __str__(self):
    # ====== get all attrs ====== #
    all_attrs = dir(self)
    padding = '  '
    print_attrs = {}
    for name in all_attrs:
      if '_' != name[0] and (len(name) >= 2 and '__' != name[:2]) and\
      'name' != name and 'is_initialized' != name:
        try:
          attr = getattr(self, name)
        except Exception:
          continue
        # check print-able type
        if is_primitives(attr, inc_ndarray=True) or \
        (hasattr(attr, '__call__') and not inspect.ismethod(attr)):
          print_attrs[name] = attr
    print_attrs = sorted(print_attrs.items(), key=lambda x: x[0])
    # ====== format the output ====== #
    ops_format = '<%s, name: %s, init: %s>\n'
    ops_format = ops_format % (ctext(self.__class__.__name__, 'cyan'),
                               ctext(self.name, 'MAGENTA'),
                               self._is_initialized)
    for i, j in print_attrs:
      if isinstance(j, NNOp): # special format for NNOp
        s = '\n' + '\n'.join([2 * padding + line
            for line in str(j).split('\n')])
      else: # other attributes
        s = dummy_formatter(j)
      ops_format += padding + "%s: %s\n" % (ctext(i, 'yellow'), s)
    # ====== print tensor ====== #
    for name, (var, vtype) in self._variable_info.items():
      if vtype != 'tensor':
        continue
      v = self.get(name)
      roles = K.role.get_roles(v)
      ops_format += padding + "(Tensor)%s shape=%s, type=%s\n, role=%s\n" % \
          (ctext(v.name.split(':')[0], 'yellow'),
           ctext(tuple(v.get_shape().as_list()), 'yellow'),
           ctext(v.dtype.base_dtype.name, 'yellow'),
           ctext(';'.join(roles), 'yellow'))
    # ====== print Variable ====== #
    for var in self.variables:
      name = var.name.split(':')[0]
      vtype = type(var).__name__
      shape = tuple(var.get_shape().as_list())
      roles = K.role.get_roles(var)
      dtype = var.dtype.base_dtype.name
      ops_format += padding + "(%s)%s shape=%s, type=%s, role=%s\n" % \
          (vtype, ctext(name, 'yellow'),
              ctext(shape, 'yellow'), ctext(dtype, 'yellow'),
              ctext(';'.join(roles), 'yellow'))
    # ====== print NNOps ====== #
    for op in self.nnops:
      name = op.name
      otype = type(op).__name__
      ops_format += padding + '(NNOp)%s type=%s, inshape=%s\n' % \
          (ctext(name, 'yellow'),
              ctext(otype, 'yellow'),
              ctext(op.input_shape, 'yellow'))
    # ====== Input info ====== #
    for i, arg in enumerate(self._args_desc):
      if isinstance(arg, VariableDesc):
        arg = str(arg)
      else:
        arg = '%s - value:%s' % (ctext(type_path(arg), 'cyan'),
                                 dummy_formatter(arg))
      name = ctext('[%d]' % i, 'yellow')
      ops_format += padding + name + arg + '\n'
    for key, arg in self._kwargs_desc.items():
      if isinstance(arg, VariableDesc):
        arg = str(arg)
      else:
        arg = 'type:%s, value:%s' % (ctext(type_path(arg), 'cyan'),
                                     dummy_formatter(arg))
      name = ctext('[%s]' % key, 'yellow')
      ops_format += padding + name + arg + '\n'
    return ops_format[:-1]

  # ==================== Slicing ==================== #
  def __getitem__(self, key):
    return NNSliceOp(self, key)


_PRIMITIVE_TYPES = (tuple, list, dict, string_types, type(True),
                    types.FunctionType, numbers.Number, type(None),
                    K.rand.constant, NNOp, VariableDesc, type)


# ===========================================================================
# Helper
# ===========================================================================
class Lambda(NNOp):

  """
  Parameters
  ----------
  func: callable
      must be picklable, main lambda function
  funcT: callable, None
      function used in transpose, if None, if the original `func`
  var_init: dict
      mapping from name (string) to variable initialization
      information.
      the initialization information can be given in 2 forms:
      - initializer: callable, ndarray, shape, Tensor, Variable, or
      string (name of Variable)
      - (initializer, roles): same as the first one but created
      variable will be assigned given roles

  Note
  ----
  There are 2 ways to feed argument for `func`:
   - by calling this Lambda
   >>> f = Lambda(func=lambda x, y=1, z=2: x + y + z)
   >>> f(1, z=3)
   - predefine the variable using `var_init`
   >>> f = Lambda(func=lambda x, y=1, z=2: x + y + z, var_init={'x': 1})
   >>> f()
  """
  @staticmethod
  def search(name, path=None, prefix='model'):
    """ This method search for any objects decorated with `Lambda`
    from given `path` with all script have given `prefix`
    """
    # ====== check path ====== #
    possible_path = ['.', './models', './model', './.models', './.model']
    script_path = os.path.dirname(sys.argv[0])
    if path is None:
      path = [os.path.join(script_path, p) for p in possible_path]
      path = [p for p in path if os.path.exists(p) and os.path.isdir(p)]
    elif not isinstance(path, (tuple, list)):
      path = [path]
    if len(path) == 0:
      raise ValueError("Cannot find any available directory that contain the "
                       "model script.")
    # ====== search for model ====== #
    for p in path:
      model_func = get_module_from_path(name, path = p, prefix = prefix)
      model_func = [f for f in model_func if isinstance(f, Lambda)]
    if len(model_func) == 0:
      raise ValueError("Cannot find any model creator function with name=%s "
                       "at paths=%s." % (name, ', '.join(path)))
    return model_func[0]

  def __init__(self, func, funcT=None, var_init={}, **kwargs):
    super(Lambda, self).__init__(**kwargs)
    # check main function
    self.set_function(func, is_transpose=False)
    # check transpose function
    if funcT is None:
      funcT = func
    self.set_function(funcT, is_transpose=True)
    # check vars
    self.var_init = {str(k): v for k, v in var_init.items()}

  def set_function(self, func, is_transpose=False):
    if not hasattr(func, '__call__'):
      raise ValueError("func must be call-able for Lambda.")
    if not isinstance(func, FuncDesc):
      func = func if is_pickleable(func) else functionable(func)
      func = FuncDesc(func)
    if is_transpose:
      self._funcT = func
    else:
      self._func = func
    return self

  def _initialize(self):
    for name, info in self.var_init.items():
      if isinstance(info, (tuple, list)) and len(info) == 2:
        init, roles = info
      else:
        init = info
        roles = []
      self.get_variable(name=name, initializer=init, roles=roles)

  def _apply(self, *args, **kwargs):
    # ====== update additional specialized variable for this NNOp ====== #
    for name in self._variable_info.keys():
      if name not in kwargs:
        kwargs[name] = self.get(name)
    return self._func(*args, **kwargs)

  def _transpose(self):
    return Lambda(func=self._funcT, funcT=self._func,
                  var_init={name: var
                            for name, (var, vtype) in self._variable_info.items()})


class NNSliceOp(NNOp):

  def __init__(self, ops, slice):
    if not isinstance(ops, NNOp):
      raise ValueError('ops must be instance of NNOp, but was given argument '
                       'has %s' % str(type(ops)))
    super(NNSliceOp, self).__init__()
    self._ops = ops
    if not isinstance(slice, (tuple, list)):
      slice = [slice]
    self.slice = slice

  def _apply(self, X, **kwargs):
    y = self._ops.apply(X, **kwargs)
    return_list = True if isinstance(y, (tuple, list)) else False
    # apply slice and calculate the shape
    output = [i[self.slice] for i in as_tuple(y)]
    # return output
    if return_list:
      return output
    return output[0]

  def __str__(self):
    ops_format = '<ops: %s, name: %s, init: %s, slice: %s>'
    return ops_format % (self._ops.__class__.__name__, self._ops.name,
                         self._ops.is_initialized, str(self.slice))


# ===========================================================================
# Simple ops
# ===========================================================================
class Dense(NNOp):

  def __init__(self, num_units,
               W_init=K.rand.glorot_uniform,
               b_init=K.rand.constant(0),
               activation=K.linear,
               **kwargs):
    super(Dense, self).__init__(**kwargs)
    self.activation = (K.linear if activation is None else activation)
    self.W_init = W_init
    self.b_init = b_init
    self.num_units = num_units

  # ==================== abstract methods ==================== #
  def _transpose(self):
    # create the new dense
    return Dense(num_units=self.input_shape[-1],
                 W_init=Lambda(func=tf.transpose,
                               var_init={'a': self.get('W')}),
                 b_init=None if self.b_init is None else 0.,
                 activation=self.activation)

  def _initialize(self):
    input_shape = self.input_shape
    shape = (input_shape[-1], self.num_units)
    self.get_variable(name='W', shape=shape, initializer=self.W_init,
                      roles=Weight)
    if self.b_init is not None:
      self.get_variable(initializer=self.b_init,
          shape=(self.num_units,), name='b', roles=Bias)

  def _apply(self, X):
    # calculate projection
    activation = K.dot(X, self.get('W'))
    # add the bias
    if self.b_init is not None:
      activation = activation + self.get('b')
    # Nonlinearity might change the shape of activation
    return self.activation(activation)


class ParametricRectifier(NNOp):
  """ This class is adpated from Lasagne:
  Original work Copyright (c) 2014-2015 lasagne contributors
  All rights reserved.
  LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE
  A layer that applies parametric rectify activation to its input
  following [1]_ (http://arxiv.org/abs/1502.01852)
  Equation for the parametric rectifier linear unit:
  :math:`\\varphi(x) = \\max(x,0) + \\alpha \\min(x,0)`
  Parameters
  ----------
  incoming : a :class:`Layer` instance or a tuple
      The layer feeding into this layer, or the expected input shape
  alpha : shared variable, expression, numpy array or call-able
      Initial value, expression or initializer for the alpha values. The
      shape must match the incoming shape, skipping those axes the alpha
      values are shared over (see the example below).
      See :func:`lasagne.utils.create_params` for more information.
  shared_axes : 'auto', 'all', int or tuple of int
      The axes along which the parameters of the rectifier units are
      going to be shared. If ``'auto'`` (the default), share over all axes
      except for the second - this will share the parameter over the
      minibatch dimension for dense layers, and additionally over all
      spatial dimensions for convolutional layers. If ``'all'``, share over
      all axes, which corresponds to a single scalar parameter.
  **kwargs
      Any additional keyword arguments are passed to the `Layer` superclass.
   References
  ----------
  .. [1] K He, X Zhang et al. (2015):
     Delving Deep into Rectifiers: Surpassing Human-Level Performance on
     ImageNet Classification,
     http://link.springer.com/chapter/10.1007/3-540-49430-8_2
  Notes
  -----
  The alpha parameter dimensionality is the input dimensionality minus the
  number of axes it is shared over, which matches the same convention as
  the :class:`BiasLayer`.
  >>> layer = ParametricRectifierLayer((20, 3, 28, 28), shared_axes=(0, 3))
  >>> layer.alpha.get_value().shape
  (3, 28)
  """

  def __init__(self, alpha_init=K.rand.constant(0.25),
               shared_axes='auto', **kwargs):
    super(ParametricRectifier, self).__init__(**kwargs)
    self.alpha_init = alpha_init
    self.shared_axes = shared_axes

  # ==================== abstract methods ==================== #
  def _initialize(self):
    if self.shared_axes == 'auto':
      self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
    elif self.shared_axes == 'all':
      self.shared_axes = tuple(range(len(self.input_shape)))
    elif isinstance(self.shared_axes, int):
      self.shared_axes = (self.shared_axes,)

    shape = [size for axis, size in enumerate(self.input_shape)
             if axis not in self.shared_axes]
    if any(size is None for size in shape):
      raise ValueError("ParametricRectifierLayer needs input sizes for "
                       "all axes that alpha's are not shared over.")
    self.alpha = self.get_variable(initializer=self.alpha_init,
        shape=shape, name="alpha", roles=Parameter)

  def _apply(self, X):
    axes = iter(range(K.ndim(self.alpha)))
    pattern = ['x' if input_axis in self.shared_axes
               else next(axes)
               for input_axis in range(K.ndim(x))]
    alpha = K.dimshuffle(self.alpha, pattern)
    return K.relu(x, alpha)
