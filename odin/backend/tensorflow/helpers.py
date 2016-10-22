"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division

import os
import warnings
import numbers
from numbers import Number
from contextlib import contextmanager
from collections import OrderedDict
from itertools import chain

import numpy as np

import tensorflow as tf
from tensorflow import variable_scope

from odin.basic import (add_role, has_roles,
                        add_shape, get_shape,
                        TRAINING, DEPLOYING,
                        AUXILIARY, PARAMETER)
from odin.utils.decorators import singleton
from odin.utils import dict_union, as_shape_tuple
from odin.config import CONFIG

FLOATX = CONFIG.floatX
NPROCESSORS = CONFIG['device_info']['n']

# ===========================================================================
# Initialize session
# ===========================================================================
# with tf.Session() as sess:
#   with tf.device("/gpu:1"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     product = tf.matmul(matrix1, matrix2

_SESSION = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=NPROCESSORS,
    allow_soft_placement=True))


def set_session(session):
    global _SESSION
    _SESSION = session


def get_session():
    return _SESSION


# ===========================================================================
# Shape helpers
# ===========================================================================
def _unique(seq, key=None):
    """ Copyright (c) 2013 Matthew Rocklin

    Return only unique elements of a sequence

    >>> tuple(unique((1, 2, 3)))
    (1, 2, 3)
    >>> tuple(unique((1, 2, 1, 3)))
    (1, 2, 3)

    Uniqueness can be defined by key keyword

    >>> tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len))
    ('cat', 'mouse')

    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for item in seq:
            if item not in seen:
                seen_add(item)
                yield item
    else:  # calculate key
        for item in seq:
            val = key(item)
            if val not in seen:
                seen_add(val)
                yield item


def auto_infer_shape(ops, *var, **kwargs):
    """ You can set 'group_inputs' in kwargs so the inputs to ops
    will be ops(var) instead of ops(*var)
    """
    try:
        inputs = []
        for i in var:
            if isinstance(i, numbers.Number):
                inputs.append(i)
            else:
                input_shape = (0 if s is None or (isinstance(s, Number) and s < 0)
                               else s
                               for s in get_shape(i))
                inputs.append(T.alloc(0, *input_shape))
        if 'group_inputs' in kwargs:
            del kwargs['group_inputs']
            output_shape = ops(inputs, **kwargs).shape.eval()
        else:
            output_shape = ops(*inputs, **kwargs).shape.eval()
        return tuple(s if s else None for s in output_shape)
    except theano.gof.MissingInputError:
        return 'None'


# ===========================================================================
# Basic query
# ===========================================================================
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
    return (not variable.owner and
            not isinstance(variable, SharedVariable) and
            not isinstance(variable, Constant))


def is_trainable_variable(variable):
    """Check if a variable is a Theano shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    return (isinstance(variable, SharedVariable) and
            not isinstance(variable, RandomStateSharedVariable) and
            not hasattr(variable.tag, 'is_rng'))


def is_variable(variable):
    """ a variable is any tensor variable in (e.g. placeholder,
    trainable_variable, intermediate tensor, ...)
    """
    return isinstance(variable, Variable)


def is_training(v):
    """ A variable is in TRAINING mode if at least one of its inputs has
    training role.

    Note
    ----
    TRAINING role can be override by: ``add_role(x, DEPLOYING)``

    """
    if not isinstance(v, (tuple, list)):
        v = [v]
    inputs = graph.inputs(v)
    for i in inputs:
        if has_roles(i, TRAINING, exact=True):
            return True
    return False


# ===========================================================================
# VARIABLE MANIPULATION
# ===========================================================================
_CREATED_VARIABLE = {}
# var id start from 0 and increasing to make sure no duplicate variable
_VAR_ID = 0


def variable(value, dtype=FLOATX, name=None, target=None):
    '''Instantiates a tensor.

    # Arguments
        value: numpy array, initial value of the tensor.
        dtype: tensor type.
        name: optional name string for the tensor.

    # Returns
        Tensor variable instance.
    '''
    variable = tf.Variable(value, dtype=dtype, name=name)
    if tf.get_default_graph() is _SESSION.graph:
        _SESSION.run(variable.initializer)
    else:
        raise Exception("The default tensorflow session have not been associated "
                        "with ODIN session, hence, cannot initialized the variable."
                        "Consider using set_session() to manually assign current "
                        "ODIN session.")
    add_shape(variable, tuple(variable.get_shape().as_list()))
    return variable


def placeholder(shape, dtype=FLOATX, name=None, for_training=False):
    shape = as_shape_tuple(shape)
    # ====== Modify add name prefix ====== #
    placeholder = tf.placeholder(dtype, shape=shape, name=name)
    if for_training:
        add_role(placeholder, TRAINING)
    else:
        add_role(placeholder, DEPLOYING)
    # store the predefined shape of placeholder
    add_shape(placeholder, shape)
    return placeholder


def as_tensor_variable(x, name=None, dtype=None):
    if dtype is None:
        dtype = x.dtype
    x = tf.convert_to_tensor(x, name=name, dtype=dtype)
    return x


def constant(value, dtype=None, shape=None, name='Const'):
    x = tf.constant(value, dtype=dtype, shape=shape, name=name)
    add_shape(x, x.get_shape())
    return x
