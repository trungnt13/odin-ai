from __future__ import division, absolute_import, print_function

import os
import re
import sys
import time
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
from odin.autoconfig import randint
from odin.utils import (as_tuple, as_list, uuid, cache_memory, is_number,
                        is_string, is_path, is_primitives, ctext,
                        flatten_list, get_all_files, is_pickleable,
                        FuncDesc, dummy_formatter, type_path,
                        get_module_from_path, wprint)
from odin.backend.role import (add_roles, has_roles, Parameter, Weight, Bias,
                               TrainableParameter, NNOpOutput)
from odin.nnet.base_desc import VariableDesc

import tensorflow as tf
from tensorflow.python.ops import init_ops

# ===========================================================================
# Other helpers
# ===========================================================================
def _get_vars_footprint(vars):
  return ';'.join([v.name for v in vars])

# ===========================================================================
# Global NNOp manager
# ===========================================================================
def get_all_nnops(scope=None, op_type=None):
  """ Return a dictionary of (name, nnops) for all created NNOp """
  allops = list(NNOp._ALL_NNOPS.values())
  # ====== matching the name scope ====== #
  if scope is not None:
    if not is_string(scope):
      scope = scope.name
    allops = [o for o in allops if o.name[:len(scope)] == scope]
  # ====== matching the NNOp type ====== #
  if op_type is not None:
    op_type = [i for i in as_tuple(op_type)
               if is_string(op_type) or issubclass(op_type, NNOp)]
    allops = [o for o in allops
              if any(o.__class__.__name__ == t if is_string(t)
                     else isinstance(o, t)
                     for t in op_type)]
  # ====== sorted by created time (earlier first) ====== #
  allops = sorted(allops, key=lambda x: x.timestamp,
                  reverse=False)
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
    raise RuntimeError("Cannot created NNOp with duplicated name, another NNOp "
                       "of type: %s, and name: '%s' has already existed" %
                       (type(NNOp._ALL_NNOPS[name]), name))
  NNOp._ALL_NNOPS[name] = nnop

# ===========================================================================
# Arguments scope
# ===========================================================================
__ARGS_SCOPE_STACK = [defaultdict(dict)]

def get_args_scope():
  return __ARGS_SCOPE_STACK[-1].copy()

@contextmanager
def args_scope(*ops_kwargs, **kwargs):
  """Stores the default arguments for the given set of applied_nnops.

  For usage, please see examples at top of the file.

  Parameters
  ----------
  ops_kwargs : series of list or tuple
    Contain the mapping from (`list_or_single_NNOp`, `**kwargs`)
     - The `list_or_single_NNOp` can be represented by string (name
       of a specific NNOp, or name the NNOp class), class (instance
       of NNOp), object (instance of any class inherited NNOp),
       list or tuple (which is the list of all above option).
     - `**kwargs` is list or dictionary containing the tuple of
       (keyword, value) that will define the defaults for each op in
       the listed ops
  kwargs : **kwargs
    default keywoard arguments for all given `NNOp` in `ops_kwargs`

  Return
  ------
  the current_scope, which is a dictionary of {op: {arg: value}}

  Raises
  ------
  TypeError: if list_ops is not a list or a tuple.
  ValueError: if any op in list_ops has not be decorated with @add_arg_scope.

  Note
  ----
  if the name scope is given, an incremental ID is generated for
  duplicated NNOp instead of UUID.
  """
  new_scope = defaultdict(dict)
  for ops, kw in ops_kwargs:
    if isinstance(kw, (tuple, list)):
      kw = dict(kw)
    if not isinstance(ops, (tuple, list)):
      ops = [ops]
    for o in ops:
      new_scope[o].update(kw)
  # ====== update Arguments Scope ====== #
  # copy prevous scopes
  args_scope = defaultdict(dict)
  for i, j in __ARGS_SCOPE_STACK[-1].items():
    args_scope[i] = j.copy()
  # update new scopes
  for op, kw in new_scope.items():
    args_scope[op].update(kw)
  # add the default kwargs
  for kw in args_scope.values():
    kw.update(kwargs)
  __ARGS_SCOPE_STACK.append(args_scope)
  # ====== return the scope ====== #
  yield None
  # ====== reset everything ====== #
  __ARGS_SCOPE_STACK.pop()

# ===========================================================================
# NNOp scope
# ===========================================================================
# each element is a list [scope_name, current_id]
_NNOP_SCOPE_STACK = []

def get_nnop_scope():
  # each element is a list [scope_name, current_id]
  if len(_NNOP_SCOPE_STACK) == 0:
    return ['', uuid()]
  return _NNOP_SCOPE_STACK[-1]

@contextmanager
def nnop_context(scope, reuse=None):
  """ A more generalized version of `tensorflow.variable_scope`,
  this function will also set the NNOp scope so any NNOp created within
  given scope will be affected

  Parameters
  ----------
  scope: string
      the name of current scope, new NNOp will be created as "scope/name"
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
  # ====== prepare Name Scope ====== #object
  curr_scope, curr_opID = get_nnop_scope()
  # NO duplicate scope
  if scope not in curr_scope:
    curr_scope = scope if len(curr_scope) == 0 else \
        curr_scope + '/' + scope
  # name scope
  _NNOP_SCOPE_STACK.append([curr_scope, 0])
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
# Main Ops
# ===========================================================================
class _NNOp_Meta(ABCMeta):
  """ This meta-class ensure the NNOp argument scope is applied

  Note
  ----
  you can only modify the arguments and kwarguments using __call__
  from MetaClass, not __new__ of instance class.
  """
  def __new__(mcs, name, bases, class_dict):
    private = {'T', 'apply', '__call__', '__getstate__', '__setstate__',
               '__getnewargs__', 'get', 'get_variable_nnop', '__setattr__',
               'input_shape', 'placeholders', 'last_output', 'apply'}
    if name != 'NNOp':
      for attr in private:
        if attr in class_dict:
          raise RuntimeError("[Class:%s]The behavior of NNOp is "
              "restricted to ensure properly operations, the following "
              "methods or properties cannot be override: '%s'" %
              (ctext(name, 'red'), ctext(attr, 'yellow')))
    return super().__new__(mcs, name, bases, class_dict)

  def __call__(clazz, *args, **kwargs):
    assert issubclass(clazz, NNOp), \
    "NNOpMeta should only be used for NNOp subclass"
    # getting the default arguments to check user intentionally override
    # default argument.
    sign = inspect.signature(clazz.__init__)
    # ignore the self argument
    default_args = OrderedDict([(n, p.default)
                                for i, (n, p) in enumerate(sign.parameters.items())
                                if i > 0 and
                                p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                               inspect.Parameter.VAR_KEYWORD)])
    # ====== update the current argument scope ====== #
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
    # This will call NNOp.__new__ to create an instance of NNOP,
    # if it found a duplicated NNOp pre-defined, it will return
    # the defined NNOp instead.
    op = clazz.__new__(clazz, *[], **new_kwargs)
    if not hasattr(op, '_name'):
      raise ValueError("NNOp must be given a name when initialized.")
    # check if op already initialized
    if op.name not in NNOp._ALL_NNOPS:
      clazz.__init__(op, *[], **new_kwargs)
    return op

@add_metaclass(_NNOp_Meta)
class NNOp(NNOpOutput):
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

  @classmethod
  def search(clazz, name, path=None, prefix='model'):
    """ This method search for any objects decorated or instance
    of given NNOp
    from given `path` with all script have given `prefix`

    Parameters
    ----------
    name : string
        specific name of finding object

    path : {string, list of string}
        single folder or list of folder (or file) to search
        for the desire module

    prefix : string
        prefix for filtering .py file
    """
    # ====== check path ====== #
    if path is None:
      possible_path = ['.', './models', './model', './.models', './.model']
      script_path = os.path.dirname(sys.argv[0])
      path = [os.path.join(script_path, p) for p in possible_path]
      path = [p for p in path if os.path.exists(p) and os.path.isdir(p)]
    elif not isinstance(path, (tuple, list)):
      path = [path]
    if len(path) == 0:
      raise ValueError("Cannot find any available directory that contain the "
                       "model script.")
    # ====== search for model ====== #
    all_errors = {}
    for p in path:
      model_func, errors = get_module_from_path(name, path=p, prefix=prefix,
                                               return_error=True)
      all_errors.update(errors)
      model_func = [f for f in model_func
                    if isinstance(f, clazz)]
    if len(model_func) == 0:
      print(
          ctext("The following Exception happened during loading the modules:",
            'lightred'))
      for fpath, error in all_errors.items():
        print(" ", fpath, ":", ctext(error, 'red'))
      raise ValueError("Cannot find any model creator function with name='%s' "
                       "at paths='%s'" % (name, '; '.join(path)))
    return model_func[0]

  def __new__(clazz, *args, **kwargs):
    """ This __new__ ensures no NNOp with duplicated name and type is
    created, hence, if it found a duplicated one, it will return the
    duplicated """
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
    # ====== For first time create instance ====== #
    # update Op name if it is None
    name = kwargs.get('name', None)
    op_scope = get_nnop_scope()
    # ====== special case Lambda op ====== #
    if clazz == Lambda:
      func = kwargs['func']
      assert inspect.isfunction(func) or inspect.ismethod(func),\
      "func for odin.nnop.base.Lambda must be a callable function or method"
      full_path = [i
                   for i in func.__qualname__.split('.')
                   if '<locals>' not in i]
      func_scope = full_path[:-1]
      func_name = full_path[-1]
      if name is None:
        name = func_name + '_' + str(op_scope[1])
      else:
        name = name
    # ====== general case ====== #
    else:
      # automatic generate name
      if name is None:
        name = clazz.__name__ + '_' + str(op_scope[1])
      # regulation for the NNOp name
      elif is_string(name):
        if '/' in name or ':' in name:
          raise ValueError("NNOp cannot contain '\\' or ':', given name is: %s" % name)
      # exception no support for given type
      else:
        raise ValueError("`name` for NNOp must be string, function, but given "
                         "`name` with type: %s" % name)
    # ====== add the scope ====== #
    # add scope to name
    if len(op_scope[0]) > 0:
      name = op_scope[0] + '/' + name
      op_scope[1] += 1
    # ====== check duplicated Op name ====== #
    if name in NNOp._ALL_NNOPS:
      old_clazz = NNOp._ALL_NNOPS[name].__class__
      if clazz != old_clazz:
        raise RuntimeError("Found predefined NNOp with type: %s, but "
            "the new NNOp has type: %s" % (old_clazz, clazz))
      return NNOp._ALL_NNOPS[name]
    # ====== allocate new Op ====== #
    created_time = time.time()
    new_op = super(NNOp, clazz).__new__(clazz)
    new_op._name = name
    new_op._timestamp = created_time
    # this store spontaneous args and kwargs feed to apply()
    new_op._current_args = ()
    new_op._current_kwargs = {}
    # all save-able attributes of NNOp store here
    new_op._save_states = {'_name': name,
                           '_timestamp': created_time}
    return new_op

  def __init__(self, **kwargs):
    # mapping: name -> VariableDesc, or Primitives
    self._kwargs_desc = OrderedDict()
    # mapping: ','.join(id(tensor)) -> output
    self._cache_outputs = {}
    self._last_input_footprint = ''
    self._transpose_ops = None
    self._is_initialized = False
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
    self._set_restore_info(None, False)

  # ==================== variable restoring ==================== #
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

  # ==================== properties ==================== #
  @property
  def save_states(self):
    """ Save state is dictionary of attribute name -> object
    those will be saved during pickling

    Note
    ----
    This property return a copy of the dictionary, any
    modification won't take effect on NNOp
    """
    return dict(self._save_states)

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
      return self.get_variable_nnop(name=name)
    # ====== nnop ====== #
    for op in self.nnops:
      if name == op.name:
        return op
    raise ValueError("Cannot find `NNOp` with name: '%s'" % name)

  @cache_memory
  def get_variable_nnop(self, name, shape=None, initializer=None, roles=[]):
    """ Initialize and return the Variable or NNOp with given description

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
      return add_roles(var, roles)
    #####################################
    # 2. initializing function.
    create_new_var = False
    if is_string(initializer):
      var = K.get_all_variables(name=initializer)
      if len(var) == 0:
        var = K.get_all_tensors(name=initializer)
      if len(var) == 0:
        raise ValueError("Cannot find any variable or tensor with name: "
            "'%s' for the initializer." % initializer)
      var = var[0]
    # is instance of NNOp
    elif isinstance(initializer, NNOp):
      var = initializer
    # is a callable
    elif hasattr(initializer, '__call__'):
      var = initializer(shape)
      if isinstance(var, np.ndarray) or\
      isinstance(initializer, init_ops.Initializer):
        create_new_var = True
    # is a scalar
    elif is_number(initializer):
      var = np.full(shape=shape, fill_value=initializer, dtype='float32')
      create_new_var = True
    # numpy array
    elif isinstance(initializer, np.ndarray):
      var = initializer
      create_new_var = True
    # actual tensor
    else:
      var = initializer
    #####################################
    # 3. Numpy ndarray.
    if create_new_var:
      var = K.variable(var, shape=shape, name=name)
      self._variable_info[name] = (var.name, 'variable')
    #####################################
    # 4. Shared variable, just check the shape.
    elif K.is_variable(var):
      _shape = var.shape.as_list()
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
      if shape is not None and var.shape.ndims != len(shape):
        raise Exception("parameter with name=%s has %d dimensions, should be "
                        "%d" % (name, var.shape.ndims, len(shape)))
      self._variable_info[name] = ((var.name, K.get_operation_footprint(var.op)),
                                   'tensor')
    elif isinstance(var, NNOp):
      self._variable_info[name] = (var, 'nnop')
    #####################################
    # 6. Exception.
    else:
      print(initializer)
      raise RuntimeError("cannot initialize parameters "
                         "(name:%s - shape:%s - roles: %s): "
                         "the `initializer` is not a numpy array, "
                         "a Tensor expression, a call-able, "
                         "or variable name as string (given type: %s)" %
                         (name, shape, roles, str(type(initializer))))
    # ====== assign annotations ====== #
    if K.is_tensor(var):
      return add_roles(var, roles)
    elif isinstance(var, NNOp):
      return var
    else:
      raise ValueError("Unsupport for variable type: %s" %
          type(var).__name__)

  @property
  def name(self):
    return self._name

  @property
  def timestamp(self):
    return self._timestamp

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

  @property
  def variables(self):
    """ Get all variables related to this Op, which include:

     - Initialized Variables
     - Variables belong to related NNOp within this Op.
     - Variables belong to related Tensor within this Op.
     - Variables within the scope of this NNOp.

    Note
    ----
    initialize all variables (basically, all Variables
    within this NNOp scope are initialized when you call
    `NNOp.variables`.
    """
    # ====== # restore variable first ====== #
    self._restore_variables()
    # ====== get all variables ====== #
    global_vars = {v.name: v for v in K.get_all_variables()}
    all_vars = []
    tensors = []
    for alias, (name, vtype) in self._variable_info.items():
      if vtype == 'variable':
        all_vars.append(global_vars[name])
      elif vtype == 'tensor':
        tensors.append(self.get_variable_nnop(alias))
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
    # exception ignore variable with name IsTraining__
    all_vars = [v for v in all_vars if 'IsTraining__' not in v.name]
    return all_vars

  @property
  def nb_variables(self):
    n = 0
    for p in self.variables:
      n += np.prod(p.shape.as_list()).astype('int32')
    return n

  @property
  def parameters(self):
    """ return all TensorVariables which have the PARAMETER role"""
    return [i for i in self.variables if has_roles(i, Parameter)]

  @property
  def trainable_parameters(self):
    """ return all TensorVariables which have the TrainableParameter role"""
    return [i for i in self.variables if has_roles(i, TrainableParameter)]

  @property
  def nb_parameters(self):
    n = 0
    for p in self.parameters:
      n += np.prod(p.shape.as_list()).astype('int32')
    return n

  @property
  def is_initialized(self):
    return self._is_initialized

  @property
  def variable_info(self):
    return dict(self._variable_info)

  @property
  def nnops(self):
    """ Return all NNOp belong to
      * the initialization of this Op
      * or within the scope of this Op.
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
    # ====== remove duplicate NNOp ====== #
    final_ops = []
    for o in ops:
      if o is not self and o not in final_ops:
        final_ops.append(o)
    return final_ops

  @property
  def placeholders(self):
    """ Create list of placeholder to represent inputs of this NNOp
    """
    x = [i.placeholder for i in self._kwargs_desc.values()
         if isinstance(i, VariableDesc)]
    return x[0] if len(x) == 1 else x

  def set_placeholder(self, name, plh):
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
    x = [tuple(i.shape.as_list())
         for i in as_tuple(self.placeholders)]
    return x[0] if len(x) == 1 else x

  @property
  def input_shape_map(self):
    return {k: tuple(v.placeholder.shape.as_list())
            for k, v in self._kwargs_desc.items()
            if isinstance(v, VariableDesc)}

  @property
  def output_shape(self):
    """NOTE: this input shape is only inferred from last inputs to
    this NNOp,
    since the input argument can has default argument, this input
    shape can change after everytime you call the NNOp"""
    output = self.last_output
    extract_shape = lambda x: tuple(x.shape.as_list()) \
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
    """Validate input argument to `apply` function

    Parameters
    ----------
    x : {tensor, primitive}
      given symbol for the argument
    name : string
      name of the argument

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
      # keywords argument
      if name not in self._kwargs_desc:
        self._kwargs_desc[name] = desc
      curr_desc = self._kwargs_desc[name]
      # validating
      if isinstance(curr_desc, VariableDesc) and not curr_desc.is_equal(desc):
        raise ValueError("Found stored argument with description: '%s', given new "
                         "argument with description: '%s'" % (str(curr_desc), str(desc)))
      # overriding primitive
      else:
        self._kwargs_desc[name] = desc
    # ====== if given raw data, use saved tensor with new data ====== #
    elif isinstance(x, np.ndarray):
      # keywords
      if name not in self._kwargs_desc:
        self._kwargs_desc[name] = VariableDesc(
            shape=x.shape, dtype=x.dtype,
            name='VarArg%d' % int(name[1:]) if '.' == name[0] else name)
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
      self._kwargs_desc[name] = x
      desc = x
    # ====== Uknown input, ERROR ====== #
    else:
      raise ValueError("The input argument for NNOp can be: "
          "`Tensor`, `odin.nnet.VariableDesc`, and primitive types"
          " (string, number, boolean, None, numpy.ndarray, numpy.generic)."
          " But the given type is: name='%s' : value=`%s`)" %
          (name, str(x)))
    return (desc, data)

  def apply(self, *args, **kwargs):
    # ====== restore variable first ====== #
    self._restore_variables()
    # ====== self.name can contain Model varable scope, hence,
    # remove the scope here  ====== #
    op_name = self.name.split('/')[-1]
    with nnop_context(scope=op_name, reuse=self.is_initialized):
      # ====== processing argument information ====== #
      # auto omit `self` argument
      sign = inspect.signature(self._apply)
      arg_name = []
      default_kwargs = {}
      inc_var_pos = False
      inc_var_key = False
      for n, p in sign.parameters.items():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
          inc_var_pos = True
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
          inc_var_key = True
        else:
          arg_name.append(n)
          if p.default != inspect.Parameter.empty:
            default_kwargs[n] = p.default
      num_args = len(arg_name)
      # adding kwargs_new in Order
      kwargs_new = OrderedDict()
      # varargs arguments is named with '.' at the beginning
      pos_name = ['.%d' % i for i in range(len(args) - num_args)] if inc_var_pos else []
      # kwargs name
      key_name = [name for name in kwargs.keys() if name not in arg_name] if inc_var_key else []
      # get all positional arguments
      for idx, name in enumerate(arg_name + pos_name + key_name):
        if idx < len(args):
          x = args[idx]
        elif name in kwargs:
          x = kwargs[name]
        elif name in default_kwargs:
          x = default_kwargs[name]
        else: # not specified, use saved arguments
          # wprint('Cannot find argument at index: %d, name: %s' % (idx, name))
          continue
        # validate and record name to self._kwargs_desc
        kwargs_new[name] = self._check_input_arg(x=x, name=name)
      # kwargs now contains: arg_name -> (VariableDesc, ndarray(or None))
      kwargs = kwargs_new
      # add missing slot from _kwargs_desc
      for name, var in self._kwargs_desc.items():
        if name not in kwargs:
          kwargs[name] = (var, None)
      # ====== create footprint for unique arguments identification ====== #
      # footprint created by concat argument name and its
      # object python ID (primitive arugment using str)
      footprint = ''
      # positional argument
      for name, (desc, dat) in sorted(kwargs.items(),
                                   key=lambda x: x[0]):
        footprint += name + ':'
        if isinstance(desc, VariableDesc): # Tensor types
          desc = desc.placeholder
          footprint += type_path(desc) + '_' + str(id(desc))
        else: # primitive types
          footprint += type_path(desc) + '_' + str(desc)
        footprint += '|'
      # ====== convert all information to new op args and kwargs ====== #
      # store current arguments
      self._current_args = []
      included_names = []
      # positional args
      for name in arg_name:
        desc, dat = kwargs[name]
        self._current_args.append(desc if is_primitives(desc, inc_ndarray=False) else
                                  desc.placeholder)
        included_names.append(name)
      # varargs
      for name in sorted([i for i in kwargs.keys() if '.' == i[0]],
                         key=lambda x: int(x[1:])):
        desc, dat = kwargs[name]
        self._current_args.append(desc if is_primitives(desc, inc_ndarray=False) else
                                  desc.placeholder)
        included_names.append(name)
      # kwargs
      self._current_kwargs = {
          name: desc if is_primitives(desc, inc_ndarray=False) else desc.placeholder
          for name, (desc, dat) in kwargs.items()
          if name not in included_names}
      given_data = {desc.placeholder: dat
                    for name, (desc, dat) in kwargs.items()
                    if isinstance(desc, VariableDesc) and dat is not None}
      # ====== initialize first ====== #
      if not self._is_initialized:
        # call NNOp initialize
        self._initialize()
        self._is_initialized = True
        # only assign new NNOp if it is initialized
        _assign_new_nnop(self)
      # ====== calculate and return outputs ====== #
      # automatically restore all variable within this NNOp scope
      self._restore_variables()
      # Recall cached output
      if footprint in self._cache_outputs:
        y = self._cache_outputs[footprint]
      else: # First time generate output given footprint
        y = self._apply(*self._current_args, **self._current_kwargs)
        # all roles to all outputs
        y = add_roles(variables=y, roles=self.__class__)
        # record cahced return
        self._cache_outputs[footprint] = y
      # check if op_data given, then evaluate to get the results.
      if len(given_data) > 0:
        # only need to make sure all variables initialized
        # if we need to evaluate some expressions.
        K.initialize_all_variables()
        y = K.eval(y, feed_dict=given_data)
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
           ctext(tuple(v.shape.as_list()), 'yellow'),
           ctext(v.dtype.base_dtype.name, 'yellow'),
           ctext(';'.join(roles), 'yellow'))
    # ====== print Variable ====== #
    for var in self.variables:
      name = var.name.split(':')[0]
      vtype = type(var).__name__
      shape = tuple(var.shape.as_list())
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
                    init_ops.Initializer, NNOp, VariableDesc, type)

# ===========================================================================
# Helper
# ===========================================================================
def _prepend_scope_nnop_tree(scope, op, parent=None):
  """ Add scope to the left of all uninitialized NNOp and its children

  This must be called at initialization, not during applying the NNOp
  """
  _ = op.name.split('/')
  op_name = _[-1]
  op_scope = _[:-1]
  op_children = op.nnops
  # ====== merge the scope ====== #
  scope = scope.split('/')
  final_scope = []
  # re-organize the scope of the new NNOp
  # - No duplicate scope
  # - prioritize scope of parent NNOp appeared first
  for s in scope + op_scope:
    if s not in final_scope:
      final_scope += [s]
  # no empty scope
  op_scope = '/'.join([i for i in final_scope if len(i) > 0])
  # ====== modification is possible ====== #
  if not op.is_initialized:
    new_name = op_scope + '/' + op_name
    # check duplicated NNOp already defined and initialized
    if new_name in NNOp._ALL_NNOPS:
      new_op = NNOp._ALL_NNOPS[new_name]
      if type(new_op) != type(op):
        raise RuntimeError("NNOp of type %s with name '%s' already defined,"
          "but given new NNOp with type %s" % (type(new_op), new_name, op))
      op = new_op
      # if `parent` is provided, modify parent NNOp list as well
      if parent is not None:
        parent._variable_info[op_name] = (new_op, 'nnop')
    # otherwise, just modify the name of newly created NNOp
    else:
      op._name = new_name
  # ====== NNOp already initialized, no change ====== #
  else:
    pass
  # ====== modify the children NNOp as well ====== #
  for o in op_children:
    _prepend_scope_nnop_tree(op_scope, o, parent=op)
  return op

class Container(NNOp):
  """ Container is NNOp that contain other NNOp """

  def __init__(self, **kwargs):
    super(Container, self).__init__(**kwargs)
    self.debug = 0

  def set_debug(self, debug_mode):
    """
    Parameters
    ----------
    debug_mode : {0, 1, 2}
        0 - turn of logging
        1 - print minimal logging
        2 - print detail logging of each NNOp
    """
    self.debug = int(debug_mode)
    return self

  def set_nnops(self, ops):
    # remove None values
    if isinstance(ops, (tuple, list)):
      ops = [o for o in ops if o is not None]
      ops = flatten_list(ops, level=None)
    ops = list(as_tuple(ops, t=NNOp))

    # add new NNOp using it name and ignore the scope
    for o in ops:
      # modify the name and scope
      o = _prepend_scope_nnop_tree(scope=self.name, op=o)
      # store the new NNOp
      self.get_variable_nnop(name=o.name.split('/')[-1], initializer=o)
    # final assignment
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
                "-> %s" % ctext(op.output_shape, 'yellow'))
        elif self.debug >= 2:
          print(str(op))
    # ====== ending and return ====== #
    if self.debug > 0:
      print('**************** End: %s ****************' %
            ctext(self.name, 'cyan'))

  def _print_op(self, op):
    # print after finish the op at each step
    self._debug_ops.append(op)

class Lambda(NNOp):

  """
  Parameters
  ----------
  func : callable
      main lambda function

  """

  def __init__(self, func, name=None):
    super(Lambda, self).__init__()
    # check main function
    self.set_function(func, is_transpose=False)
    self.set_function(func, is_transpose=True)

  def set_function(self, func, is_transpose=False):
    if not hasattr(func, '__call__'):
      raise ValueError("func must be call-able for Lambda.")
    if not isinstance(func, FuncDesc):
      func = FuncDesc(func)
    if is_transpose:
      self._funcT = func
    else:
      self._func = func
    return self

  def _initialize(self):
    pass

  def _apply(self, *args, **kwargs):
    # ====== update additional specialized variable for this NNOp ====== #
    for name in self._variable_info.keys():
      if name not in kwargs:
        kwargs[name] = self.get(name)
    return self._func(*args, **kwargs)

  def _transpose(self):
    return Lambda(name=self._funcT)

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
               W_init=init_ops.glorot_uniform_initializer(seed=randint()),
               b_init=init_ops.constant_initializer(0),
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
    self.get_variable_nnop(name='W', shape=shape, initializer=self.W_init,
                      roles=Weight)
    if self.b_init is not None:
      self.get_variable_nnop(initializer=self.b_init,
          shape=(self.num_units,), name='b', roles=Bias)

  def _apply(self, X):
    # calculate projection
    activation = K.dot(X, self.get('W'))
    # add the bias
    if self.b_init is not None:
      activation = activation + self.get('b')
    # Nonlinearity might change the shape of activation
    return self.activation(activation)
