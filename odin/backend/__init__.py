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


def variable(value=None, shape=None, dtype=floatX, name=None, roles=[]):
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
    if tf.get_default_graph() is get_session().graph:
        get_session().run(variable.initializer)
    else:
        raise Exception("The default tensorflow session have not been associated "
                        "with ODIN session, hence, cannot initialized the variable."
                        "Consider using set_session() to manually assign current "
                        "ODIN session.")
    return role.add_role(variable, roles)


def placeholder(shape=None, dtype=floatX, name=None, roles=[]):
    if shape is None and name is None:
        raise ValueError("shape and name arguments cannot be None at the same time.")
    # ====== check duplicated placeholder ====== #
    if name is not None:
        all_placeholders = [o._outputs[0] for o in get_operations(type='Placeholder')]
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
    cPickle.dump([collections, var_meta],
                 open(path + '.collections', 'w'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    return checkpoint


def restore_variables(path, session=None):
    if session is None:
        session = get_session()
    # ====== load and check var meta ====== #
    collections, var_meta = cPickle.load(open(path + '.collections', 'r'))
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


def _eval_single_tensor(x, feed_dict=None):
    if hasattr(x, 'eval') and inspect.ismethod(x.eval):
        if 'feed_dict' in inspect.getargspec(x.eval).args:
            return x.eval(session=get_session(), feed_dict=feed_dict)
        else:
            return x.eval(session=get_session())
    elif is_string(x):
        return builtins.eval(x)
    elif isinstance(x, tf.Operation):
        return get_session().run(x, feed_dict=feed_dict)
    raise ValueError("Type %s don't have the eval function." % str(x))


def eval(x, feed_dict=None):
    '''Evaluates the value of a tensor.
    Parameters
    ----------
    x: list, tuple, dictionary, `Tensor`
        tensorfow `Tensor` for evaluation
    '''
    if isinstance(x, (tuple, list)):
        return [_eval_single_tensor(tensor, feed_dict=feed_dict)
                for tensor in x]
    elif isinstance(x, Mapping):
        return {name: _eval_single_tensor(tensor, feed_dict=feed_dict)
            for name, tensor in x.iteritems()}
    return _eval_single_tensor(x, feed_dict=feed_dict)
