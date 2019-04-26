"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division

import re
import warnings
from six import string_types
from contextlib import contextmanager
from collections import OrderedDict, defaultdict, Mapping

import numpy as np

# ====== tensorflow and tensorflow-probability package ====== #
# I am sick of ImportWarning
with warnings.catch_warnings():
  import tensorflow as tf

  _tf_distribution_types = []
  warnings.filterwarnings('ignore', category=ImportWarning)
  try:
    from tensorflow.contrib import distributions as tfd
    _tf_distribution_types.append(tfd.Distribution)
  except ImportError:
    pass
  except AttributeError:
    pass

  try:
    from tensorflow_probability import distributions as _tfd
    _tf_distribution_types.append(_tfd.Distribution)
  except ImportError:
    pass
  except AttributeError:
    pass

  _tf_distribution_types = tuple(_tf_distribution_types)

# ====== O.D.I.N stuffs ====== #
from odin.autoconfig import get_session
from odin.utils.cache_utils import cache_memory
from odin.utils import (dict_union, as_list, flatten_list, as_tuple, is_string,
                        decorators, batching, Progbar)
from odin.backend.role import (has_roles, Auxiliary, Parameter)

# ===========================================================================
# Helper
# ===========================================================================
_TF_SCOPE_PATTERN = lambda scope, begin: \
re.compile(r"%s%s(_\d+)?\/" % ('^' if bool(begin) else '', scope))

# ===========================================================================
# Basic query
# ===========================================================================
def is_training(graph=None):
  if graph is None:
    graph = get_session().graph
  training_var = get_all_variables(scope=None, name='IsTraining__',
                                   graph=graph)
  if len(training_var) == 0:
    raise RuntimeError("Cannot find variable with name='IsTraining' scope='' "
                       "within graph=%s" % str(graph))
  elif len(training_var) > 1:
    raise RuntimeError("Found multiple 'IsTraining__' flag: %s" %
      str(training_var))
  return training_var[0]

def set_training(is_training, graph=None, return_ops=False):
  if graph is None:
    graph = get_session().graph
  return set_value(is_training(graph), bool(is_training),
                   return_ops=return_ops)

def cond_training(train_fn, infer_fn,
                  train_dependencies=None, infer_dependencies=None,
                  strict=False, graph=None,
                  name=None):
  """ Create a conditional output that depends of training variable

  Parameters
  ----------
  train_fn: callable
    this function will be called when `training=True`
  infer_fn: callable
    this function will be called when `training=False`
  train_dependencies: Tensorflow expression
    all of given expression will be executed before running
    the function
  infer_dependencies: Tensorflow expression
    all of given expression will be executed before running
    the function
  strict: bool (default: False)
    if False, automatically group all outputs into single tensorflow
    Tensor, this behaviour is disabled with `strict=True`
  name: str
    specific name for the Operation

  """
  if not hasattr(train_fn, '__call__') or \
  not hasattr(infer_fn, '__call__'):
    raise ValueError("`train_fn` and `infer_fn` must be call-able.")
  with tf.name_scope(name, "TrainingCondition"):
    # ====== training ====== #
    if train_dependencies is not None:
      def _train_fn():
        with tf.control_dependencies(as_tuple(train_dependencies)):
          return train_fn()
    else:
      _train_fn = train_fn
    # ====== inference ====== #
    if infer_dependencies is not None:
      def _infer_fn():
        with tf.control_dependencies(as_tuple(infer_dependencies)):
          return infer_fn()
    else:
      _infer_fn = infer_fn
    return tf.cond(pred=is_training(graph=graph),
                   true_fn=_train_fn, false_fn=_infer_fn,
                   strict=strict)

def is_operation(op):
  return isinstance(op, tf.Operation)

def is_placeholder(variable):
  """Check if variable is a user-provided graph input.

  To be considered an input the variable must have no owner, and not
  be a constant or shared variable.

  Parameters
  ----------
  variable : :class:`~tensor.TensorVariable`

  Returns
  -------
  bool
      ``True`` If the variable is a user-provided input to the graph.

  """
  return isinstance(variable, tf.Tensor) and \
      variable.op.node_def.op == "Placeholder"

def is_variable(variable):
  """Check if a variable is a shared variable.

  Notes
  -----
  This function excludes shared variables that store the state of Theano
  random number generators.

  """
  if (isinstance(variable, tf.Variable) and
          variable.op.node_def.op == "VariableV2"):
    return True
  return False

def is_distribution(x):
  return isinstance(x, _tf_distribution_types)

def is_tensor(variable, inc_distribution=True, inc_variable=True):
  """ a variable is any tensor variable in (e.g. placeholder,
  trainable_variable, intermediate tensor, ...)
  All `TensorType` includes:
  * ops.Tensor
  * sparse_tensor.SparseTensor
  * variables.Variable
  """
  _ = tf.contrib.framework.is_tensor(variable)
  if not inc_variable:
    _ &= (not isinstance(variable, tf.Variable))
  if inc_distribution:
    _ |= is_distribution(variable)
  return _

def set_shape(tensor, shape):
  """ This function will filling the missing shape information
  of given tensor
  """
  if not is_tensor(tensor):
    raise ValueError('tensor must be instance of `Tensor`.')
  # ====== Test ====== #
  ndims = tensor.shape.ndims
  shape = as_tuple(shape)
  if ndims != len(shape):
    raise ValueError("The tensor has %d dimensions, but the given shape "
                     "has %d dimension." % (ndims, len(shape)))
  # ====== DO it ====== #
  old_shape = tensor.shape
  new_shape = []
  for old, new in zip(old_shape, shape):
    old_value = old.value
    if isinstance(new, tf.Dimension):
      new = new.value
    # matching old and new values
    if old_value is not None and new is not None:
      if old_value != new:
        raise ValueError("Known shape information mismatch, from tensorflow"
            ":%s, and given shape:%s." %
            (str(old_shape.as_list()), str(shape)))
      else:
        new_shape.append(old_value)
    elif old_value is None and new is not None:
      new_shape.append(new)
    elif old_value is not None and new is None:
      new_shape.append(old_value)
    elif old is None and new is None:
      new_shape.append(old)
    else:
      new_shape.append(None)
  tensor.set_shape(new_shape)
  return tensor

# ===========================================================================
# VALUE MANIPULATION
# ===========================================================================
def get_value(x):
  if isinstance(x, (tuple, list)):
    return get_session().run(x)
  return x.eval(session=get_session())

def set_value(x, value, return_ops=False, name='SetValue'):
  '''Sets the value of a tensor variable,
  from a Numpy array.

  Parameters
  ----------
  x: `Tensor`
  value: real value
  return_ops: bool
      if True, return assign Op and feed_dict instead of running
      the Op directly
  '''
  if isinstance(value, np.ndarray):
    value = value.astype(x.dtype.as_numpy_dtype)
  elif is_tensor(value):
    value = tf.cast(value, dtype=x.dtype)
  assign_op = tf.assign(x, value, name=name)
  if return_ops:
    return assign_op
  get_session().run(assign_op)
  return x

def get_epsilon(x_or_dtype):
  """ Return the epsilon given a dtype or dtype of given Tensor

  Parameters
  ----------
  x_or_dtype : {numpy.ndarray, tensorflow.Tensor, numpy.dtype, string}
      anything that contain the given dtype information
  """
  if isinstance(x_or_dtype, string_types):
    dtype = np.dtype(x_or_dtype)
  elif hasattr(x_or_dtype, 'dtype'):
    dtype = x_or_dtype.dtype
    if hasattr(dtype, 'as_numpy_dtype'):
      dtype = dtype.as_numpy_dtype
  else:
    raise ValueError("Cannot infer dtype from: %s" %
      str(type(x_or_dtype)))
  return np.finfo(dtype).eps

# ===========================================================================
# Session helper
# ===========================================================================
def _filter_string(criterion, x):
  if isinstance(criterion, string_types):
    return criterion == x
  elif hasattr(criterion, '__call__'):
    return criterion(x)
  elif criterion is None:
    return True
  raise ValueError("Unknown criterion for filtering.")

_ops_ID = {}
_name_pattern = re.compile(r".*_\d+")

def get_normalized_name(var, shape=True, dtype=True):
  """Get normalized name of `Tensor`, `Variable` or `Op` in tensorflow
  The normalized name remove:
   * '_%d' suffix of scope and name.
   * ':%d' indices of name.

  shape: if True, append string of shape to name
  dtype: if True, append string of dtype to name
  """
  x = var.name
  # remove the index
  x = x.split(':')[0]
  # remove the _%d
  x = '/'.join(['_'.join(i.split('_')[:-1]) if _name_pattern.match(i) else i
                for i in x.split('/')])
  if shape:
    x += str(tuple(var.shape.as_list())).replace(' ', '')
  if dtype:
    x += np.dtype(var.dtype.as_numpy_dtype).name
  return x

def get_all_operations(otype=None, device=None, sort=False, scope=None,
                       footprint=None, graph=None, beginning_scope=True):
  """ Return list of all operations in default graph
  The follow attributes can be access within the operation:
   * name : string
   * otype : string, operation type (e.g. `"MatMul"`).
   * device:  string name of the device to which this op has been assigned
   * _inputs : list of `Tensor`
   * _outputs : list of `Tensor`
   * _control_inputs : Before this op is executed, the operations in
       `control_inputs` have finished executing.
   * graph : `Graph` that contains this operation
   * node_def : serialized `NodeDef` representation of this operation.
   * op_def : `OpDef` proto that represents the type of this op.
   * traceback : call stack from when this operation was constructed.

  Some important op type:
   * "Placeholder"
   * "VariableV2"
   * "Const"
   * "Assign"

  Parameters
  ----------
  beginning_scope : bool (default: True)
    if True, the provide scope must be the beginning scope,
    otherwise, it could be in the middle of multiple scopes
  """
  if graph is None:
    graph = get_session().graph
  ops = graph.get_operations()
  # update OpID
  if len(_ops_ID) != len(ops):
    for ID, op in graph._nodes_by_id.items():
      if op not in _ops_ID:
        _ops_ID[op] = ID
  # filter out some op
  if otype is not None:
    ops = [o for o in ops
           if _filter_string(otype, o.type)]
  if device is not None:
    ops = [o for o in ops
           if _filter_string(device, o.device)]
  # ====== filter by scope ====== #
  if scope is not None:
    scope = str(scope)
    if len(scope) == 0:
      ops = [o for o in ops
             if '/' not in o.name]
    else:
      scope_name_pattern = _TF_SCOPE_PATTERN(scope, beginning_scope)
      ops = [o for o in ops
             if len(scope_name_pattern.findall(o.name))]
  # ====== filter by unique footprint ====== #
  if footprint is not None:
    ops = [o for o in ops
           if get_operation_footprint(o) == footprint]
  # sorted by OpID
  if sort and len(ops) > 1:
    ops = sorted(ops, key=lambda x: _ops_ID[x])
  return ops

def get_operationID(op, graph=None):
  """operation ID is unique ID of Op, the ID represent the order
  of created Op."""
  if graph is None:
    graph = get_session().graph
  ops = graph.get_operations()
  # update OpID
  if len(_ops_ID) != len(ops):
    for ID, op in graph._nodes_by_id.items():
      if op not in _ops_ID:
        _ops_ID[op] = ID
  return _ops_ID[op]

@cache_memory
def get_operation_footprint(op):
  """ Trace back the inputs of given Op and record all:
  * placholders
  * variables
  * ops
  Those are related to given op.

  The final footprint is concatenated string of all variables,
  placeholders, constants, and Ops

  Note
  ----
  This is just a fair attempt to create short identification of a
  tenorflow Op
  still have chance for duplication op footprint
  """
  if not isinstance(op, tf.Operation) and hasattr(op, 'op'):
    op = op.op
  op_id = str(op.name) + str(op.type)
  var = []
  placeholder = []
  const = []
  ops = [op.type]
  inputs = list(op.inputs)
  while len(inputs) > 0:
    i = inputs.pop()
    o = i.op
    ops.append(o.type)
    if o.type == "VariableV2":
      var.append(i)
    elif o.type == "Placeholder":
      placeholder.append(i)
    elif o.type == "Const":
      const.append(i)
    inputs = list(o.inputs) + inputs
  return op_id + '|' +\
  ':'.join([get_normalized_name(v) for v in var]) + '|' +\
  ':'.join([get_normalized_name(p) for p in placeholder]) + '|' +\
  ':'.join([get_normalized_name(c) for c in const]) + '|' +\
  ':'.join([j.split(':')[0] for j in ops])

def get_all_variables(scope=None, name=None, full_name=None,
                      graph_keys=[tf.GraphKeys.GLOBAL_VARIABLES,
                                  tf.GraphKeys.LOCAL_VARIABLES,
                                  tf.GraphKeys.MODEL_VARIABLES,
                                  tf.GraphKeys.TRAINABLE_VARIABLES],
                      graph=None, beginning_scope=True):
  """
  Parameters
  ----------
  scope: {str, None}
      scope name which the Variables have been created
  name: str
      name of tensor (WITHOUT variable scope)
  full_name: str
      name of tensor WITH variable scope.
  beginning_scope : bool (default: True)
    if True, the provide scope must be the beginning scope,
    otherwise, it could be in the middle of multiple scopes
  """
  var = []
  # ====== first get all available variable ====== #
  for k in graph_keys:
    if graph is None:
      var += [i for i in tf.get_collection(k)
              if isinstance(i, tf.Variable)]
    else:
      var += [i for i in graph.get_collection(k)
              if isinstance(i, tf.Variable)]
  var = list(set(var))
  # filtering: start from general to detail
  # ====== filter by scope ====== #
  if scope is not None:
    scope = str(scope)
    if len(scope) == 0:
      var = [v for v in var
             if '/' not in v.name]
    else:
      scope_name_pattern = _TF_SCOPE_PATTERN(scope, beginning_scope)
      var = [v for v in var
             if len(scope_name_pattern.findall(v.name))]
  # ====== filter by name ====== #
  if name is not None:
    name = as_tuple(name, t=string_types)
    var = [v for v in var
           if any((v.name.split('/')[-1] == n or
                   v.name.split('/')[-1].split(':')[0] == n.split(':')[0])
                  for n in name)]
  # ====== filter by fullname ====== #
  if full_name is not None:
    full_name = as_tuple(full_name, t=string_types)
    var = [v for v in var
           if any((n == v.name or
                   n.split(':')[0] == v.name.split(':')[0])
                  for n in full_name)]
  return var

def get_all_tensors(scope=None, name=None, full_name=None, device=None):
  """
  Parameters
  ----------
  scope: {str, None}
      scope name which the Variables have been created
  name: str
      name of tensor (without variable scope)
  full_name: str
      name of tensor WITH variable scope.
  device : {str, None}
      name of the device to which this op has been assigned
      (e.g. /cpu:0, or /gpu:0)
  """
  ops = get_all_operations(device=device, scope=scope, sort=False)
  alltensors = []
  for o in ops:
    alltensors += list(o.inputs) + list(o._outputs)
    for i in o.control_inputs:
      alltensors += list(i.inputs) + list(i._outputs)
  alltensors = list(set(alltensors))
  # ====== filter out unsupport types ====== #
  if name is not None:
    name = as_tuple(name, t=string_types)
    alltensors = [t for t in alltensors
                  if any((n == t.name.split('/')[-1] or
                          n.split(':')[0] == t.name.split('/')[-1].split(':')[0])
                         for n in name)]
  if full_name is not None:
    full_name = as_tuple(full_name, t=string_types)
    alltensors = [t for t in alltensors
                  if any((n == t.name or
                          n.split(':')[0] == t.name.split(':')[0])
                        for n in full_name)]
  return alltensors

def get_all_variables_or_tensors(scope=None, name=None, full_name=None):
  var = get_all_variables(scope=scope, name=name, full_name=full_name)
  if len(var) == 0:
    var = get_all_tensors(scope=scope, name=name, full_name=full_name)
  return var

# ===========================================================================
# ComputationGraph
# ===========================================================================
class Function(object):
  """ Two way to call this Function
  f(x1, x2, x3) or f('x1'=x1, 'x2'=x2, 'x3'=x3)

  Parameters
  ----------
  inputs: list of `tf.placeholder` or `tf.Variable`

  outputs: list of `tf.Tensor`

  updates: list, or dict
      mapping from `Tensor` to its new value which is `Tensor` or
      real value.

  defaults: dict
      mapping from `Variable` or `placeholder` to its default values.

  training: None, True, False
      if `training=None`, left the training mode unchanged
      if `training=True`, turn on training mode only when execute this
      function.
      if `training=False`, disable training mode only when execute this
      function.

  batch_size : {int, None} (default: None)
      if `batch_size` is not None, auto-split all array into minibatch,
      and return a list of outputs (all array must have equal `shape[0]`)

  batch_vars : {Tensor, list of Tensor}
      if `len(batch_vars) == 0`, split mini-batches for all Tensor inputs,
      otherwise, only applying for a selected set of inputs.

  strict : bool (default: True)
      if False, remove shape mis-matched inputs when `__call__`
      this `Function`.
      if True, raise RuntimeError.

  Note
  ----
  call the function with `show_progress=True` when `batch_size`
  is specified to show the progress.

  """

  def __init__(self, inputs, outputs, updates=[], defaults={},
               training=None, batch_size=None, batch_vars=[],
               strict=False):
    self.training = training
    self.batch_size = batch_size
    self.batch_vars = list(batch_vars)
    self._strict = bool(strict)
    # ====== validate input ====== #
    if isinstance(inputs, Mapping):
      self.inputs_name = inputs.keys()
      inputs = inputs.values()
    elif not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    self.inputs = flatten_list(inputs, level=None)
    if not hasattr(self, 'inputs_name'):
      self.inputs_name = [i.name.split(':')[0] for i in self.inputs]
    # ====== defaults ====== #
    defaults = dict(defaults)
    self.defaults = defaults
    # ====== validate outputs ====== #
    return_list = True
    if not isinstance(outputs, (tuple, list)):
      outputs = (outputs,)
      return_list = False
    self.outputs = flatten_list(list(outputs), level=None)
    self._return_list = return_list
    # ====== validate updates ====== #
    if isinstance(updates, Mapping):
      updates = updates.items()
    with tf.control_dependencies(self.outputs):
      # create updates ops
      if not isinstance(updates, tf.Operation):
        updates_ops = []
        for update in updates:
          if isinstance(update, (tuple, list)):
            p, new_p = update
            updates_ops.append(tf.assign(p, new_p))
          else: # assumed already an assign op
            updates_ops.append(update)
        self.updates_ops = tf.group(*updates_ops)
      else: # already an tensorflow Ops
        self.updates_ops = updates
    # ====== cached shape ====== #
    self._input_shape = [tuple(i.shape.as_list()) for i in self.inputs]
    self._output_shape = [tuple(i.shape.as_list()) for i in self.outputs]

  @property
  def input_shape(self):
    return self._input_shape

  @property
  def output_shape(self):
    return self._output_shape if self._return_list else self._output_shape[0]

  def __call__(self, *inputs, **kwargs):
    show_progress = kwargs.pop('show_progress', False)
    # dictionary as inputs
    if len(kwargs) == len(self.inputs_name):
      inputs = [kwargs[i] for i in self.inputs_name]
    # ====== delete un-matchede inputs ====== #
    inputs_new = []
    tmp = list(inputs)
    shapes = list(self._input_shape)
    # this process iteratively remove inputs with mismatch shape
    # to current given input
    for s in shapes:
      for i in tuple(tmp):
        if len(i.shape) != len(s) or \
        any(a is not None and a > 0 and a != b
                for a, b in zip(s, i.shape)): # different ndim, or shape
          tmp.remove(i)
        else:
          inputs_new.append(i)
          tmp.remove(i)
          break
    if len(inputs_new) != len(self.inputs):
      raise ValueError("Given inputs have shape: %s, cannot match the shape of "
                       "defined inputs: %s" %
                       ('; '.join([str(i.shape) for i in inputs]),
                        '; '.join([str(i) for i in self.input_shape])))
    if not self._strict:
      inputs = inputs_new
    # ====== create feed_dict ====== #
    feed_dict = {}
    inputs = flatten_list(inputs, level=None)
    for tensor, value in zip(self.inputs, inputs):
      feed_dict[tensor] = value
    feed_dict.update(self.defaults)
    # check if modifying training mode
    if self.training is None:
      pass
    elif self.training:
      feed_dict.update({is_training(): True})
    else:
      feed_dict.update({is_training(): False})
    session = get_session()
    outputs = None
    # ====== mini-batches ====== #
    if self.batch_size is not None:
      batch_vars = ([i for i in feed_dict.keys() if is_tensor(i)]
                    if len(self.batch_vars) == 0 else self.batch_vars)
      batch_vars = [i for i in batch_vars
                    if i in feed_dict and hasattr(feed_dict[i], 'shape')]
      n_samples = list(set(feed_dict[i].shape[0] for i in batch_vars))
      assert len(n_samples) == 1, \
      "Data have multiple batching dimension: %s" % str(n_samples)
      n_samples = n_samples[0]
      # only continue if we have more samples than `batch_size`
      if n_samples > self.batch_size:
        n_output = len(self.outputs)
        outputs = []
        all_batches = []
        # (optional) showing progress
        if show_progress:
          prog = Progbar(target=n_samples,
                         print_report=False, print_summary=False,
                         name='')
        for s, e in batching(batch_size=int(self.batch_size),
                             n=n_samples):
          if show_progress:
            prog.add(e - s)
          all_batches.append(e - s)
          feed_dict_minibatch = OrderedDict([(k, v[s:e])
                                             if k in batch_vars else (k, v)
                                             for k, v in feed_dict.items()])
          updated = session.run(self.outputs + [self.updates_ops],
                                feed_dict=feed_dict_minibatch)
          updated = updated[:n_output]
          if not self._return_list:
            updated = updated[0]
          outputs.append(updated)
        ## concatenate all outputs
        if not self._return_list:
          o_ndim = outputs[0].ndim
          if o_ndim == 0: # returned scalars
            outputs = np.array(outputs)
          else: # returned array
            for o_axis in range(o_ndim):
              all_n = [o.shape[o_axis] for o in outputs]
              if all_n == all_batches:
                break
            outputs = np.concatenate(outputs, axis=o_axis)
        ## returning a list of outputs
        else:
          new_outputs = []
          for output_idx in range(len(outputs[0])):
            o = [x[output_idx] for x in outputs]
            o_ndim = o[0].ndim
            if o_ndim == 0: # returned scalars
              o = np.array(o)
            else: # returned array
              for o_axis in range(o[0].ndim):
                all_n = [val.shape[o_axis] for val in o]
                if all_n == all_batches:
                  break
              o = np.concatenate(o, axis=o_axis)
            new_outputs.append(o)
          outputs = new_outputs
    # ====== single batch ====== #
    if outputs is None:
      updated = session.run(self.outputs + [self.updates_ops],
                            feed_dict=feed_dict)
      outputs = updated[:len(self.outputs)]
      if not self._return_list:
        outputs = outputs[0]
    # ====== return final output ====== #
    return outputs

def function(inputs, outputs, updates=[], defaults={},
             training=None, batch_size=None, batch_vars=[]):
  """
  Parameters
  ----------
  inputs: list of `tf.placeholder` or `tf.Variable`
  outputs: list of `tf.Tensor`
  updates: list, or dict
      mapping from `Tensor` to its new value which is `Tensor` or
      real value.
  defaults: dict
      mapping from `Variable` or `placeholder` to its default values.
  training: None, True, False
      if `training=None`, left the training mode unchanged
      if `training=True`, turn on training mode only when execute this
      function.
      if `training=False`, disable training mode only when execute this
      function.
  batch_size : {int, None} (default: None)
      if `batch_size` is not None, auto-split all array into minibatch,
      and return a list of outputs (all array must have equal `shape[0]`)
  batch_vars : {Tensor, list of Tensor}
      if `len(batch_vars) == 0`, split mini-batches for all Tensor inputs,
      otherwise, only applying for a selected set of inputs.
  """
  # ====== check inputs ====== #
  if inputs is None or len(as_tuple(inputs)) == 0:
    inputs = ComputationGraph(outputs).placeholders
    inputs_text = ', '.join([str(i) for i in inputs])
    print("[WARNING] inputs haven't specified, auto-inferred from Graph of "
          "outputs, graph inputs: %s" % 'None' if len(inputs) == 0 else inputs_text)
  return Function(inputs=inputs, outputs=outputs,
                  updates=updates, defaults=defaults,
                  training=training, batch_size=batch_size)

# ===========================================================================
# Computational graph
# ===========================================================================
def get_intermediate_tensors(outputs, scope=None, name=None, full_name=None,
                             roles=None, match_all=False, exact=False,
                             beginning_scope=False):
  """ Return all variables and tensor within the computational
  graph of the `outputs` that are satisfied the given conditions

  Parameters
  ----------
  scope : {None, string}
    name of variable scope, any of the scope that match given name
    will be selected
  name : {None, string}
    the name of tensor without the output indexing ":0" and the scope
  full_name : {None, string}
    the full name includes both scope and tensor name without the
    output indexing ":0"
  roles : {None, odin.backend.role}
    specific roles of the tensor
  match_all : bool (default: False)
    If ``True``, checks if the variable has all given roles.
    If ``False``, any of the roles is sufficient.
  exact : bool (default: False)
    If ``True``, use ``==`` for comparison to get exactly same roles.
    If ``False``, use `issubclass` for comparison, hence, also match the
    descendant roles.
  beginning_scope : bool (default: False)
    if True, the provide scope must be the beginning scope,
    otherwise, it could be in the middle of multiple scopes
  """
  graph = ComputationGraph(outputs)
  return graph.get(scope=scope, name=name, full_name=full_name,
                   roles=roles, match_all=match_all, exact=exact,
                   beginning_scope=beginning_scope)

@decorators.singleton
class ComputationGraph(object):
  r"""Encapsulates a managed Theano computation graph.

  This implies that it not only contains the variables required to
  compute the given outputs, but also all the auxiliary variables and
  updates that were attached to these variables through the annotation
  system.

  All variables are presented in topologically sorted order according to
  the apply nodes that they are an input to.

  Parameters
  ----------
  outputs : (list of) :class:`~tf.Tensor`
      The output(s) of the computation graph.
  trace_up : bool
      if True, all the descendant `Tensor` that computed based on
      those outputs will be traced.

  Note
  ----
  This :class:`ComputationGraph` is a `singleton` class since once a graph
  is created, it will never be changed (node insertion or deletion).

  Attributes
  ----------
  inputs : list of :class:`~tensor.TensorVariable`
      The inputs of the computation graph. This does not include shared
      variables and constants.
  trainable_variables : list of :class:`~tensor.TensorSharedVariable`
      All the shared variables in the graph.
  parameters : list of :class:`~tensor.TensorSharedVariable`
      All the shared variables which have the :const:`.PARAMETER` role.
  outputs : list of :class:`~tensor.TensorVariable`
      The outputs of the computations graph (as passed to the
      constructor).
  auxiliary_variables : list of :class:`~tensor.TensorVariable`
      All variables which have the :const:`.AUXILIARY` role.
  intermediary_variables : list of :class:`~tensor.TensorVariable`
      Any variable that is not part of :attr:`inputs` or :attr:`outputs`.
  variables : list of :class:`~tensor.TensorVariable`
      All variables (including auxiliary) in the managed graph.
  scans : list of :class:`~theano.scan_module.scan_op.Scan`
      All Scan ops used in this computation graph.
  scan_variables : list of :class:`~tensor.TensorVariable`
      All variables of the inner graphs of Scan ops.
  updates : :class:`~tensor.TensorSharedVariable` updates
      All the updates found attached to the annotations.

  """

  @classmethod
  def _get_id(clazz, outputs=None, trace_up=False):
    return outputs

  def __init__(self, outputs=None, trace_up=False):
    # it is important to don't have duplicated outputs
    # otherwise, it can go into infinite loop
    outputs = list(set([o for o in flatten_list(as_list(outputs),
                                           level=None)
                        if o is not None]))
    self.outputs = outputs
    self._trace_up = trace_up
    self._get_variables()

  def _get_variables(self):
    """ Collect variables, updates and auxiliary variables.

    In addition collects all :class:`.Scan` ops and recurses in the
    respective inner Theano graphs. """
    _travelled_up = [] # to prevent recursive ops
    _travelled_down = [] # to prevent recursive ops

    def get_all_tensor_trace_down(x):
      """ recursively travel down the inputs tree to get all
      tensor """
      tensors = []
      op = x.op
      # ====== check travelled ops ====== #
      if op in _travelled_down:
        return variables
      else:
        _travelled_down.append(op)
      # ====== get all variable ====== #
      inputs = list(op.inputs)
      tensors += inputs
      for i in inputs:
        tensors += get_all_tensor_trace_down(i)
      return tensors

    def get_all_tensor_trace_up(x):
      """ travel up the outputs tree to get all variables"""
      tensors = []
      # ====== check travelled ops ====== #
      for op in x.consumers():
        if op in _travelled_up:
          continue
        else:
          _travelled_up.append(op)
        # ====== get all variable ====== #
        inputs = [i for i in op.inputs if i != x]
        outputs = list(op._outputs)
        tensors += inputs + outputs
        for o in outputs:
          tensors += get_all_tensor_trace_up(o)
      return tensors

    def create_tensor_iter(outputs):
      # if specific outputs is given
      if len(outputs) > 0:
        for o in outputs:
          # travese each node of graph
          all_tensors = get_all_tensor_trace_down(o)
          if self._trace_up:
            all_tensors += get_all_tensor_trace_up(o)
          for t in all_tensors:
            yield t
      # get all variables and tensor within the graph
      else:
        for o in outputs:
          with o.graph.as_default():
            for op in get_all_operations(sort=False):
              for t in list(op.inputs) + list(op._outputs):
                yield t
    # store all the updates embedded into the Tensor Variables
    variables = [o for o in self.outputs if is_variable(o)]
    outputs = [o for o in self.outputs if not is_variable(o)]
    tensors = []
    placeholders = []
    # ====== travese each node of graph ====== #
    # first get all available variables
    # (even not related to outputs)
    global_vars = {}
    if len(outputs) > 0:
      for o in outputs:
        with o.graph.as_default():
          for v in get_all_variables():
            global_vars[v.name] = v
            global_vars[v.value().name] = v
    else:
      for v in get_all_variables():
        global_vars[v.name] = v
        global_vars[v.value().name] = v
    # then iterate over all tensor
    for t in create_tensor_iter(outputs):
      if t.name in global_vars:
        variables.append(global_vars[t.name])
      elif is_tensor(t):
        tensors.append(t)
    variables = list(set(variables))
    tensors = list(set(tensors + outputs))
    # sorted by Ops ID in _nodes=
    graph_nodes_ID = {}
    for t in tensors:
      if t.graph not in graph_nodes_ID:
        graph_nodes_ID[t.graph] = {op: ID
            for ID, op in t.graph._nodes_by_id.items()}
    tensors = sorted(tensors, key=lambda x: graph_nodes_ID[x.graph][x.op])
    placeholders = [t for t in tensors if is_placeholder(t)]

    self._placeholders = placeholders
    self._tensors = tensors
    self._variables = variables

  # ==================== Get variables ==================== #
  @property
  def placeholders(self):
    """Inputs to the graph, excluding constants and shared variables."""
    return list(self._placeholders)

  @property
  def tensors(self):
    return self._tensors

  @property
  def variables(self):
    return list(self._variables)

  @property
  def parameters(self):
    return [var for var in self._variables if has_roles(var, Parameter)]

  @property
  def auxiliary_tensors(self):
    return [t for t in self._tensors if has_roles(t, Auxiliary)]

  @property
  def dict_of_placeholders(self):
    """Return a mapping from an input name to the input."""
    return {var.name.split(':')[0]: var for var in self.placeholders}

  # ==================== Graph manipulation ==================== #
  def swap(self, dict_swap, scope="copied", replace_itself=False, copy_q=False):
    """ Original implementation: Edward (https://github.com/blei-lab/edward)
    Build a new node in the TensorFlow graph from `org_instance`,
    where any of its ancestors existing in `dict_swap` are
    replaced with `dict_swap`'s corresponding value.

    The copying is done recursively, so any `Operation` whose output
    is required to evaluate `org_instance` is also copied (if it isn't
    already copied within the new scope). This is with the exception of
    `tf.Variable`s, `tf.placeholder`s, and nodes of type `Queue`, which
    are reused and not newly copied.

    Parameters
    ----------
    dict_swap : dict, optional
      Random variables, variables, tensors, or operations to swap with.
      Its keys are what `org_instance` may depend on, and its values are
      the corresponding object (not necessarily of the same class
      instance, but must have the same type, e.g., float32) that is used
      in exchange.
    scope : str, optional
      A scope for the new node(s). This is used to avoid name
      conflicts with the original node(s).
    replace_itself : bool, optional
      Whether to replace `org_instance` itself if it exists in
      `dict_swap`. (This is used for the recursion.)
    copy_q : bool, optional
      Whether to copy the replaced tensors too (if not already
      copied within the new scope). Otherwise will reuse them.

    Returns
    -------
    RandomVariable, tf.Variable, tf.Tensor, or tf.Operation
      The copied node.

    Examples
    --------
    >>> from odin import backend as K
    >>> x = tf.constant(2.0, name='x')
    >>> y = tf.constant(3.0, name='y')
    >>> z = tf.multiply(x, y, name="z")
    ... # define replacement variables
    >>> qx = tf.constant(4.0, name='qx')
    >>> qz = tf.constant(25.0, name='qz')
    ... # The TensorFlow graph is currently
    ... # `x` -> `z` <- y`, `qx`
    ... # This adds a subgraph with newly copied nodes,
    ... # `copied/qx` -> `copied/z` <- `copied/y`
    >>> z_new = K.ComputationGraph(z).swap(
    ...     dict_swap={x: qx, z: qz},
    ...     replace_itself=False, copy_q=False)
    >>> print([v.name for v in K.ComputationGraph(z_new).variables])
    ... # [u'qx:0', u'copied/y:0', u'copied/z:0', u'copied/w:0']
    >>> sess = tf.Session()
    >>> sess.run(z) # 6.0
    >>> sess.run(z_new) # 12.0
    ... # with replace_itself = True
    >>> z_new = K.ComputationGraph(z).swap(
    ...     dict_swap={x: qx, z: qz},
    ...     replace_itself=True, copy_q=False)
    >>> print([v.name for v in K.ComputationGraph(z_new).variables])
    ... # [u'qx:0', u'copied/y:0', u'qz:0', u'copied/w:0']
    >>> sess.run(z_new) # 25.0
    """
    try:
      from edward import copy
    except ImportError:
      raise RuntimeError("Require Edward library to manipulate the "
                         "ComputationGraph.")
    outputs_new = []
    for o in self.outputs:
      o_new = copy(org_instance=o, dict_swap=dict_swap, scope=scope,
               replace_itself=replace_itself, copy_q=copy_q)
      dict_swap[o] = o_new
      outputs_new.append(o_new)
    return outputs_new

  def get(self, scope=None, name=None, full_name=None,
          roles=None, match_all=False, exact=False,
          beginning_scope=False):
    """ Return all variables and tensor with given roles

    Parameters
    ----------
    scope : {None, string}
      name of variable scope, any of the scope that match given name
      will be selected
    name : {None, string}
      the name of tensor without the output indexing ":0" and the scope
    full_name : {None, string}
      the full name includes both scope and tensor name without the
      output indexing ":0"
    roles : {None, odin.backend.role}
      specific roles of the tensor
    match_all : bool (default: False)
      If ``True``, checks if the variable has all given roles.
      If ``False``, any of the roles is sufficient.
    exact : bool (default: False)
      If ``True``, use ``==`` for comparison to get exactly same roles.
      If ``False``, use `issubclass` for comparison, hence, also match the
      descendant roles.
    beginning_scope : bool (default: True)
      if True, the provide scope must be the beginning scope,
      otherwise, it could be in the middle of multiple scopes
    """
    alltensors = self.tensors + self.variables
    # ====== by role ====== #
    if roles is not None:
      alltensors = [t for t in alltensors
                    if has_roles(t, roles=roles,
                                 match_all=match_all, exact=exact)]
    # ====== from general to detail ====== #
    if scope is not None:
      scope = str(scope)
      if len(scope) == 0:
        alltensors = [t for t in alltensors
                      if '/' not in t.name]
      else:
        scope_name_pattern = _TF_SCOPE_PATTERN(scope, beginning_scope)
        alltensors = [t for t in alltensors
                      if len(scope_name_pattern.findall(t.name))]
    # ====== filter by name ====== #
    if name is not None:
      name = as_tuple(name, t=string_types)
      alltensors = [t for t in alltensors
                    if any((n == t.name.split('/')[-1] or
                            n.split(':')[0] == t.name.split('/')[-1].split(':')[0])
                           for n in name)]
    # ====== full name ====== #
    if full_name is not None:
      full_name = as_tuple(full_name, t=string_types)
      alltensors = [t for t in alltensors
                    if any((n == t.name or
                            n.split(':')[0] == t.name.split(':')[0])
                           for n in full_name)]
    return alltensors

  # ==================== others ==================== #
  def __len__(self):
    return len(self.variables)

  def __iter__(self):
    for v in self.variables:
      yield v

  def __del__(self):
    self.dispose()
    del self.outputs
    del self.variables
