from __future__ import print_function, division, absolute_import

import os
import inspect
from six import string_types
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


def get_all_variables(scope=None, name=None, full_name=None,
                      graph_keys=[tf.GraphKeys.GLOBAL_VARIABLES,
                                  tf.GraphKeys.LOCAL_VARIABLES,
                                  tf.GraphKeys.MODEL_VARIABLES,
                                  tf.GraphKeys.TRAINABLE_VARIABLES]):
    """
    Parameters
    ----------
    name: str
        name of tensor (without variable scope)
    full_name: str
        name of tensor WITH variable scope.
    """
    var = []
    for k in graph_keys:
        var += [i for i in tf.get_collection(k) if isinstance(i, tf.Variable)]
    var = list(set(var))
    if scope is not None:
        scope_name_pattern = re.compile('%s_?\d*\/' % str(scope))
        var = [v for v in var if len(scope_name_pattern.findall(v.name))]
    if name is not None:
        name = as_tuple(name, t=string_types)
        var = [v for v in var
               if any((v.name.split('/')[-1] == n or
                       v.name.split('/')[-1] == n + ':0') for n in name)]
    if full_name is not None:
        full_name = as_tuple(full_name, t=string_types)
        var = [v for v in var
               if any((n == v.name or
                       n + ':0' == v.name) for n in full_name)]
    return var


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
        all_variables = get_all_variables()
        for v in all_variables:
            v_shape = tuple(v.get_shape().as_list())
            if v.name == name: # found duplicated variable
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
    placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
    return role.add_role(placeholder, roles)


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
        return [_eval_single_tensor(x, feed_dict=feed_dict) for tensor in x]
    elif isinstance(x, dict):
        return {name: _eval_single_tensor(tensor, feed_dict=feed_dict)
            for name, tensor in x.iteritems()}
    return _eval_single_tensor(x, feed_dict=feed_dict)


_saver = {}


def save_variables(var_list, path, session=None):
    """ This function only apply for trainable parameters """
    if session is None:
        session = get_session()
    var_list = as_tuple(var_list)
    name = '|'.join(sorted([v.name for v in var_list]))
    if name in _saver:
        saver = _saver[name]
    else:
        saver = tf.train.Saver(var_list=var_list, restore_sequentially=False,
            allow_empty=False)
    # ====== save the variables ====== #
    checkpoint = saver.save(session, path, global_step=None,
        write_meta_graph=False, write_state=False)
    # ====== save the collections ====== #
    collections = {var.name: role.get_roles(var, return_string=True)
                   for var in var_list}
    cPickle.dump(collections, open(path + '.collections', 'w'),
        protocol=cPickle.HIGHEST_PROTOCOL)
    return checkpoint


def restore_variables(var_list, path, session=None):
    if session is None:
        session = get_session()
    var_list = as_tuple(var_list)
    name = '|'.join(sorted([v.name for v in var_list]))
    if name in _saver:
        saver = _saver[name]
    else:
        saver = tf.train.Saver(var_list=var_list, restore_sequentially=False,
            allow_empty=False)
    # ====== save the variables ====== #
    saver.restore(session, path)
    # ====== save the collections ====== #
    collections = cPickle.load(open(path + '.collections', 'r'))
    for var, roles in collections.iteritems():
        variable(name=var, roles=roles)
