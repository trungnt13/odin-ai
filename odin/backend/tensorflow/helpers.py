"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division

import os
import warnings
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework import is_tensor as _is_tensor
from tensorflow.contrib.distributions.python.ops.distribution import Distribution as _Distribution
from tensorflow import variable_scope as _tf_variable_scope

from odin.basic import (add_role, has_roles, as_shape_tuple, is_training,
                        add_shape, get_shape, Auxiliary, Parameter)
from odin.utils.decorators import singleton
from odin.utils import dict_union, as_list, flatten_list
from odin.config import CONFIG

FLOATX = CONFIG.floatX
NPROCESSORS = CONFIG['device_info']['n']

# ===========================================================================
# Initialize session
# ===========================================================================
__session_args = {
    'intra_op_parallelism_threads': NPROCESSORS,
    'allow_soft_placement': True,
    'log_device_placement': False,
}
if CONFIG['device'] == 'gpu':
    if CONFIG['cnmem'] > 0:
        __session_args['gpu_options'] = tf.GPUOptions(
            per_process_gpu_memory_fraction=CONFIG['cnmem'],
            allow_growth=False)
    else:
        __session_args['gpu_options'] = tf.GPUOptions(
            allow_growth=True)
_SESSION = tf.InteractiveSession(config=tf.ConfigProto(**__session_args))


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
    """Check if a variable is a shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    if (isinstance(variable, tf.Variable) and
            variable.op.node_def.op[:8] == "Variable"):
        return variable in variable.graph.get_collection('trainable_variables')
    return False


def is_variable(variable):
    """ a variable is any tensor variable in (e.g. placeholder,
    trainable_variable, intermediate tensor, ...)
    """
    return _is_tensor(variable) or isinstance(variable, _Distribution)


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


def variable_scope(scope):
    return _tf_variable_scope(scope, reuse=False)


def variable(value, dtype=FLOATX, name=None):
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
        variable = _CREATED_VARIABLE[full_name]
        if get_shape(variable) != value.shape:
            raise Exception('Found pre-defined variable with scope="%s", name="%s" '
                            'and shape="%s", but the new value has shape="%s"' %
                            (current_scope, name, get_shape(variable), value.shape))
        else:
            print("[WARNING] Load new value to the old variable with name:", full_name)
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
    placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
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

    def __init__(self, *outputs):
        outputs = flatten_list(outputs, level=None)
        self.outputs = [o for o in outputs if o is not None]
        self._get_variables()

    def _get_variables(self):
        """ Collect variables, updates and auxiliary variables.

        In addition collects all :class:`.Scan` ops and recurses in the
        respective inner Theano graphs. """
        _travelled_op = [] # to prevent recursive ops

        def get_all_variables(x):
            """ recursively travel down the inputs tree to get all
            variables """
            variables = []
            op = x.op
            # ====== check travelled ops ====== #
            if op in _travelled_op:
                return variables
            else:
                _travelled_op.append(op)
            # ====== get all variable ====== #
            inputs = op._inputs
            variables += inputs
            for i in inputs:
                variables += get_all_variables(i)
            return variables

        def create_variables_iter(outputs):
            if len(outputs) > 0:
                for o in outputs:
                    # travese each node of graph
                    for v in get_all_variables(o):
                        yield v
            else:
                graph = get_session().graph
                all_ops = graph.get_operations()
                for o in all_ops:
                    for v in o._inputs + o._outputs + o._control_inputs:
                        yield v
        # store all the updates embedded into the Tensor Variables
        updates = OrderedDict()
        shared_outputs = [o for o in self.outputs if is_trainable_variable(o)]
        usual_outputs = [o for o in self.outputs if not is_trainable_variable(o)]
        variables = shared_outputs
        trainable_variables = list(shared_outputs)
        inputs = []
        # if the list of outputs is specified
        # ====== travese each node of graph ====== #
        # first get all variables
        global_vars = {}
        if len(usual_outputs) > 0:
            for o in usual_outputs:
                for v in o.graph.get_collection('variables'):
                    global_vars[v.name] = v
        else:
            for v in get_session().graph.get_collection('variables'):
                global_vars[v.name] = v
        # then iterate over all tensor
        for v in create_variables_iter(usual_outputs):
            _travelled_op = [] # reset the tracking list
            if v.name in global_vars:
                variables.append(global_vars[v.name])
            if _is_tensor(v):
                variables.append(v)
        variables = list(set(variables + usual_outputs))
        # sorted by Ops ID in _nodes=
        graph_nodes_ID = {}
        for v in variables:
            if v.graph not in graph_nodes_ID:
                graph_nodes_ID[v.graph] = {op: ID
                    for ID, op in v.graph._nodes_by_id.iteritems()}
        variables = sorted(variables, key=lambda x: graph_nodes_ID[x.graph][x.op])
        inputs = [v for v in variables if is_placeholder(v)]
        trainable_variables = [v for v in variables if is_trainable_variable(v)]
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
                if has_roles(var, [Parameter])]

    @property
    def auxiliary_variables(self):
        return [var for var in self.variables if has_roles(var, [Auxiliary])]

    @property
    def dict_of_placeholders(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.placeholders}

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

    @contextmanager
    def with_roles(self, roles, match_all=False, exact=False):
        vars = [v for v in self.variables
                if has_roles(v, roles, match_all=match_all, exact=exact)]
        # tracking if new variables have been created
        all_graph = list(set([v.graph for v in self.variables]))
        old_max_ops = {g: g._nodes_by_id.keys()[-1] for g in all_graph}
        yield vars
        new_max_ops = {g: g._nodes_by_id.keys()[-1] for g in all_graph}
        # ====== check if new Ops is performed in this context ====== #
        for g in all_graph:
            if old_max_ops[g] < new_max_ops[g]:
                for i in range(old_max_ops[g], new_max_ops[g] + 1):
                    op = g._nodes_by_id[i]
                    # TODO: think about what to do with new Ops here

    # ==================== others ==================== #
    def __iter__(self):
        for v in self.variables:
            yield v

    def __del__(self):
        self.dispose()
        del self.outputs
        del self.variables
