from __future__ import print_function, division, absolute_import

import os
import inspect
from collections import Mapping
from six.moves import cPickle, builtins

from odin.utils import is_string, is_path, as_tuple
from odin.config import (auto_config, get_floatX, get_session,
                         get_rng, randint)
auto_config()
floatX = get_floatX()
# ==================== import utilities modules ==================== #
# from .basic_ops import *
# from .advance_ops import *
from .helpers import *
from .tensor import *
from .activations import *
from . import role
from . import metrics
from . import optimizers
from . import rand
from . import rnn_cell


def variable(value=None, shape=None, dtype=floatX, name=None, roles=[],
             initialize=False):
  '''Instantiates a tensor, automatically initialize the variable
  in tensorflow

  Parameters
  ----------
  value: numpy array
      initial value of the tensor.
  dtype: dtype
      tensor type.
  name: str
      optional name string for the tensor.
  roles: {Role, list of Role}
      given Role for initialized Variable from `odin.backend.role`
  initialize : bool
      if True, call Session run to initialize the variable.

  Returns
  -------
      Tensor variable instance.
  '''
  # check name and value
  if value is not None:
    value = np.array(value)
  #### Found cached variable, just load new value into it
  if name is not None:
    for v in get_all_variables(scope=tf.get_variable_scope().name,
                               name=name):
      v_shape = tuple(v.get_shape().as_list())
      # set new value for variable
      if (value is not None and v_shape != value.shape) or \
      (shape is not None and v_shape != as_tuple(shape)):
        raise ValueError("Pre-defined variable with name: %s and"
            " shape: %s, which is different from given shape: %s"
            % (name, v_shape,
                value.shape if value is not None else shape))
      # just get the variable
      return role.add_role(v, roles)
  #### create totally new variable
  if value is None:
    variable = tf.get_variable(name=name, shape=shape)
  else:
    if shape is not None and value.shape != tuple(shape):
      raise ValueError("Given value has shape:%s, but the given shape is"
                       ":%s." % (value.shape, shape))
    variable = tf.Variable(value, dtype=dtype, name=name)
  # initialize variable
  if initialize:
    get_session(graph=variable.graph).run(variable.initializer)
  return role.add_role(variable, roles)


def initialize_all_variables(vars=None):
  if vars is None:
    vars = get_all_variables()
  else:
    vars = [v for v in as_tuple(vars)
            if is_variable(v)]
  # ====== check if variable not initialized ====== #
  init_info = eval([tf.is_variable_initialized(v) for v in vars])
  vars = [v for v, inited in zip(vars, init_info) if not inited]
  # ====== build mapping graph -> list of vars ====== #
  graph = defaultdict(list)
  for var in get_all_variables():
    graph[var.graph].append(var)
  # ====== run the initialization ====== #
  for g, v in graph.items():
    get_session(graph=g).run([i.initializer for i in v])


def is_variable_initialized(var):
  if not is_variable(var):
    raise ValueError("`var` must be instance of tensorflow.Variable, "
                     "but given type: %s" % type(var))
  return eval(tf.is_variable_initialized(var))


def placeholder(shape=None, dtype=floatX, name=None, roles=[]):
  if shape is None and name is None:
    raise ValueError("shape and name arguments cannot be None at the same time.")
  # ====== check duplicated placeholder ====== #
  if name is not None:
    all_placeholders = [
        o._outputs[0] for o in get_all_operations(otype='Placeholder')]
    for v in all_placeholders:
      v_shape = tuple(v.get_shape().as_list())
      if v.name == name + ':0': # found duplicated variable
        # set new value for variable
        if shape is not None:
          if v_shape == shape:
            return role.add_role(v, roles)
          else:
            raise ValueError("Pre-defined placeholder with name: %s and"
                " shape: %s, which is different from given shape: %s"
                % (name, v_shape, shape))
        # just get the variable
        else:
          return role.add_role(v, roles)
  # ====== Modify add name prefix ====== #
  plh = tf.placeholder(dtype=dtype, shape=shape, name=name)
  return role.add_role(plh, roles)


_saver = {}


def save_variables(var_list, path, session=None):
  """ This function only apply for trainable parameters """
  if session is None:
    session = get_session()
  var_list = [v for v in set(as_tuple(var_list)) if is_variable(v)]
  name = '|'.join(sorted([v.name for v in var_list]))
  if name in _saver:
    saver = _saver[name]
  else:
    saver = tf.train.Saver(var_list=var_list, restore_sequentially=False,
        allow_empty=False)
  # ====== save the variables ====== #
  checkpoint = saver.save(session, path, global_step=None,
      write_meta_graph=False, write_state=False)
  # ====== save meta-info for recreate variable ====== #
  var_meta = []
  for v in var_list:
    name = v.name.split(':')[0]
    dtype = v.dtype.base_dtype.name
    shape = v.get_shape().as_list()
    var_meta.append((name, dtype, shape))
  # ====== save the collections ====== #
  collections = {var.name: role.get_roles(var, return_string=True)
                 for var in var_list}
  with open(path + '.collections', 'wb') as f:
    cPickle.dump([collections, var_meta], f,
                 protocol=cPickle.HIGHEST_PROTOCOL)
  return checkpoint


def save_graph(path, graph=None):
  g = tf.summary.FileWriter(path)
  if graph is None:
    graph = get_session().graph
  elif isinstance(graph, tf.Session):
    graph = graph.graph
  g.add_graph(graph)
  g.flush()
  g.close()


def restore_variables(path, session=None):
  if session is None:
    session = get_session()
  # ====== load and check var meta ====== #
  with open(path + '.collections', 'rb') as f:
    collections, var_meta = cPickle.load(f)
  var_list = []
  allvars = {v.name.split(':')[0]: v for v in get_all_variables()}
  for name, dtype, shape in var_meta:
    if name in allvars: # found predefined variable
      var_list.append(allvars[name])
    else: # create new variable
      if tf.get_variable_scope().name:
        raise RuntimeError("The current variable scope is: %s, you can "
            "only restore variables from default scope."
            % tf.get_variable_scope().name)
      var_list.append(tf.get_variable(
          shape=shape, name=name, dtype=dtype))
  # ====== restore the variables ====== #
  name = '|'.join(sorted([v.name for v in var_list]))
  if name in _saver:
    saver = _saver[name]
  else:
    saver = tf.train.Saver(var_list=var_list, restore_sequentially=False,
        allow_empty=False)
  saver.restore(session, path)
  # ====== restore the collections ====== #
  for v in var_list:
    role.add_role(v, collections[v.name])


def eval(x, feed_dict=None):
  '''Evaluates the value of a tensor.

  Parameters
  ----------
  x: list, tuple, dictionary, `Tensor`
      tensorfow `Tensor` for evaluation
  '''
  # ====== list of Tensor or string ====== #
  if isinstance(x, (tuple, list)):
    string_eval = []
    tensor_eval = []
    tensor_idx = []
    # evaluate string expression
    for i, j in enumerate(x):
      if is_string(j):
        string_eval.append(builtins.eval(j))
      else:
        tensor_eval.append(j)
        tensor_idx.append(i)
    # evaluate tensor
    if len(tensor_eval) > 0:
      graph = [i.graph for i in tensor_eval]
      if len(set(graph)) > 1:
        raise RuntimeError("Cannot evaluate multiple `Tensor` come from "
                           "different `Graph`.")
      tensor_eval = get_session(graph[0]).run(tensor_eval,
                                              feed_dict=feed_dict)
    return tuple([tensor_eval.pop(0) if i in tensor_idx else
                  string_eval.pop(0)
                  for i in range(len(x))])
  # ====== mapping ====== #
  elif isinstance(x, Mapping):
    results = {}
    tensor_eval_key = []
    tensor_eval_value = []
    for k, v in x.items():
      if is_string(v):
        results[k] = builtins.eval(v)
      else:
        tensor_eval_key.append(k)
        tensor_eval_value.append(v)
    # evaluate tensor
    if len(tensor_eval) > 0:
      graph = [i.graph for i in tensor_eval_value]
      if len(set(graph)) > 1:
        raise RuntimeError("Cannot evaluate multiple `Tensor` come from "
                           "different `Graph`.")
      tensor_eval_value = get_session(graph[0]).run(tensor_eval_value,
                                                    feed_dict=feed_dict)
    # update results
    for k, v in zip(tensor_eval_key, tensor_eval_value):
      results[k] = v
    return results
  # ====== just a string ====== #
  elif is_string(x):
    return builtins.eval(x)
  # ====== just a Tensorflow object ====== #
  elif isinstance(x, tf.Operation) or \
  is_tensor(x, inc_distribution=True, inc_variable=True):
    return get_session(x.graph).run(x, feed_dict=feed_dict)
  # ====== exception ====== #
  raise RuntimeError("Cannot evaluate object of type: %s" % type(x))
