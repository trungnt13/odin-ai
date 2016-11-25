"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division

import os
import warnings
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework import is_tensor
from tensorflow import variable_scope

from odin.basic import (add_role, has_roles,
                        add_shape, get_shape,
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
    return isinstance(variable, tf.Tensor) and \
        variable.op.node_def.op == "Placeholder"


def is_trainable_variable(variable):
    """Check if a variable is a Theano shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    return (isinstance(variable, tf.Variable) and
            variable.op.node_def.op == "Variable")


def is_variable(variable):
    """ a variable is any tensor variable in (e.g. placeholder,
    trainable_variable, intermediate tensor, ...)
    """
    return is_tensor(variable)


# ===========================================================================
# VALUE MANIPULATION
# ===========================================================================
def get_value(x):
    if isinstance(x, (tuple, list)):
        return get_session().run(x)
    return x.eval(session=get_session())


def set_value(x, value):
    '''Sets the value of a tensor variable,
    from a Numpy array.
    '''
    value = np.asarray(value, dtype=x.dtype.as_numpy_dtype)
    if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
    else:
        assign_placeholder = tf.placeholder(dtype=x.dtype.base_dtype, shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
    get_session().run(assign_op, feed_dict={assign_placeholder: value})


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
    # ensure unique name
    if name is None:
        global _VAR_ID; name = 'VAR_%d' % _VAR_ID; _VAR_ID += 1
    #### Found cached variable, just load new value into it
    current_scope = tf.get_variable_scope().name
    full_name = name if len(current_scope) == 0 else current_scope + '/' + name
    if full_name in _CREATED_VARIABLE:
        variable = _CREATED_VARIABLE[name]
        if get_shape(variable) != value.shape:
            raise Exception('Found pre-defined variable with shape="%s" but new'
                            ' value has shape="%s"' % (get_shape(variable), value.shape))
        else:
            warnings.warn("Load value of new variable to old variable, "
                          "var's name:" + name)
        set_value(variable, value)
        return variable
    #### create totally new variable
    variable = tf.Variable(value, dtype=dtype, name=name)
    if tf.get_default_graph() is _SESSION.graph:
        _SESSION.run(variable.initializer)
    else:
        raise Exception("The default tensorflow session have not been associated "
                        "with ODIN session, hence, cannot initialized the variable."
                        "Consider using set_session() to manually assign current "
                        "ODIN session.")
    add_shape(variable, tuple(variable.get_shape().as_list()))
    # ====== save all created variable ====== #
    _CREATED_VARIABLE[variable.name.split(':')[0]] = variable
    return variable


def placeholder(shape, dtype=FLOATX, name=None):
    shape = as_shape_tuple(shape)
    # ====== Modify add name prefix ====== #
    placeholder = tf.placeholder(dtype, shape=shape, name=name)
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


# ===========================================================================
# ComputationGraph
# ===========================================================================
@singleton
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
    outputs : (list of) :class:`~tensor.TensorVariable`
        The output(s) of the computation graph.

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

    def __init__(self, outputs):
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        self.outputs = list(outputs)
        self._get_variables()

    def _get_variables(self):
        """Collect variables, updates and auxiliary variables.

        In addition collects all :class:`.Scan` ops and recurses in the
        respective inner Theano graphs.

        """
        def get_all_variables(x):
            """ recursively travel down the inputs tree to get all
            variables """
            variables = []
            op = x.op
            inputs = op._inputs
            variables += inputs
            for i in inputs:
                variables += get_all_variables(i)
            return variables

        updates = OrderedDict()

        shared_outputs = [o for o in self.outputs if is_trainable_variable(o)]
        usual_outputs = [o for o in self.outputs if not is_trainable_variable(o)]
        variables = shared_outputs
        trainable_variables = [v for v in shared_outputs if is_trainable_variable(v)]
        inputs = []

        if usual_outputs:
            for o in self.outputs:
                trainable_collections = {i.name: i
                    for i in o.graph._collections['trainable_variables']}
                # ====== travese each node of graph ====== #
                for v in get_all_variables(o):
                    if is_placeholder(v):
                        inputs.append(v)
                    elif v.op.node_def.op == "Variable" and v.name in trainable_collections:
                        v = trainable_collections[v.name]
                        trainable_variables.append(v)
                    if is_tensor(v):
                        variables.append(v)
            inputs = list(set(inputs))
            variables = list(set(variables))
            trainable_variables = list(set(trainable_variables))
        # ====== get all updates and auxiliary variables ====== #
        for v in inputs + variables:
            if hasattr(v, 'tag'):
                # updates
                _ = getattr(v.tag, 'updates', OrderedDict())
                _ = OrderedDict([(i, j) for i, j in _.iteritems()
                                 if is_variable(i)])
                updates = dict_union(updates, _)
                # auxiliary_variables
                for _ in getattr(v.tag, 'auxiliary_variables', []):
                    if _ not in variables:
                        variables.append(_)
        self._inputs = inputs
        self.variables = variables
        self._trainable_variables = trainable_variables
        self.updates = updates

    # ==================== Get variables ==================== #
    @property
    def inputs(self):
        """ Same as placeholder """
        return self.placeholders

    @property
    def placeholders(self):
        """Inputs to the graph, excluding constants and shared variables."""
        return self._inputs

    @property
    def intermediary_variables(self):
        return [var for var in self.variables if
                var not in self.placeholders and
                var not in self.outputs]

    @property
    def trainable_variables(self):
        return self._trainable_variables

    @property
    def parameters(self):
        return [var for var in self.trainable_variables
                if has_roles(var, [PARAMETER])]

    @property
    def auxiliary_variables(self):
        return [var for var in self.variables if has_roles(var, [AUXILIARY])]

    @property
    def dict_of_placeholders(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.placeholders}

    # ==================== others ==================== #
    def __iter__(self):
        for v in self.variables:
            yield v

    def __del__(self):
        self.dispose()
        del self.outputs
        del self.variables
