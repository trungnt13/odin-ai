"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division

import re
from six import string_types
from contextlib import contextmanager
from collections import OrderedDict, defaultdict

import numpy as np

import tensorflow as tf
from tensorflow.contrib.distributions import Distribution as _Distribution

from odin.config import get_session
from odin.utils.decorators import singleton
from odin.utils import dict_union, as_list, flatten_list, as_tuple

from .role import (has_roles, Auxiliary, Parameter)


# ===========================================================================
# Basic query
# ===========================================================================
def __init_training_var():
    x = tf.Variable(initial_value=False, dtype=tf.bool, name='IsTraining',
                    trainable=False)
    get_session().run(x.initializer)
    return x
__IS_TRAINING = defaultdict(__init_training_var)


def is_training(graph=None):
    if graph is None:
        graph = get_session().graph
    return __IS_TRAINING[graph]


def set_training(is_training, graph=None, return_ops=False):
    if graph is None:
        graph = get_session().graph
    return set_value(__IS_TRAINING[graph], bool(is_training),
                     return_ops=return_ops)


def cond_training(train_fn, infer_fn,
                  train_dependencies=None, infer_dependencies=None,
                  name="ConditionalTraining"):
    with tf.variable_scope(name):
        if train_dependencies is not None:
            def _train_fn():
                with tf.control_dependencies(as_tuple(train_dependencies)):
                    return train_fn()
        else:
            _train_fn = train_fn
        if infer_dependencies is not None:
            def _infer_fn():
                with tf.control_dependencies(as_tuple(infer_dependencies)):
                    return infer_fn()
        else:
            _infer_fn = infer_fn
        return tf.cond(__IS_TRAINING, fn1=_train_fn, fn2=_infer_fn)


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
    return isinstance(x, _Distribution)


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
    ndims = tensor.get_shape().ndims
    shape = as_tuple(shape)
    if ndims != len(shape):
        raise ValueError("The tensor has %d dimensions, but the given shape "
                         "has %d dimension." % (ndims, len(shape)))
    # ====== DO it ====== #
    old_shape = tensor.get_shape()
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
        value = np.asarray(value, dtype=x.dtype.as_numpy_dtype)
    elif is_tensor(value):
        value = tf.cast(value, dtype=x.dtype.base_dtype)
    assign_op = tf.assign(x, value, name=name)
    if return_ops:
        return assign_op
    get_session().run(assign_op)
    return x


# ===========================================================================
# Session helper
# ===========================================================================
def _filter_string(criterion, x):
    if isinstance(criterion, string_types):
        return criterion == x
    elif callable(criterion):
        return criterion(x)
    elif criterion is None:
        return True
    raise ValueError("Unknown criterion for filtering.")


def get_graph():
    return get_session().graph


_ops_ID = {}


def get_operations(type=None, device=None, sort=True, scope=None):
    """ Return list of all operations in default graph
    The follow attributes can be access within the operation:
     * name : string
     * type : string, type of the op (e.g. `"MatMul"`).
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
     *
    """
    ops = get_graph().get_operations()
    # update OpID
    if len(_ops_ID) != len(ops):
        for ID, op in get_graph()._nodes_by_id.iteritems():
            if op not in _ops_ID:
                _ops_ID[op] = ID
    # filter out some op
    if type is not None:
        ops = [o for o in ops if _filter_string(type, o.type)]
    if device is not None:
        ops = [o for o in ops if _filter_string(device, o.device)]
    if scope is not None:
        scope_name_pattern = re.compile('%s_?\d*\/' % str(scope))
        ops = [o for o in ops if len(scope_name_pattern.findall(o.name))]
    # sorted by OpID
    if sort and len(ops) > 1:
        ops = sorted(ops, key=lambda x: _ops_ID[x])
    return ops


def get_operationID(op):
    ops = get_graph().get_operations()
    # update OpID
    if len(_ops_ID) != len(ops):
        for ID, op in get_graph()._nodes_by_id.iteritems():
            if op not in _ops_ID:
                _ops_ID[op] = ID
    return _ops_ID[op]


def get_tensors(name=None, full_name=None, device=None, scope=None):
    """
    Parameters
    ----------
    name: str
        name of tensor (without variable scope)
    full_name: str
        name of tensor WITH variable scope.
    """
    ops = get_operations(device=device, scope=scope, sort=False)
    alltensors = []
    for o in ops:
        alltensors += o._inputs + o._outputs
        for i in o._control_inputs:
            alltensors += i._inputs + i._outputs
    alltensors = list(set(alltensors))
    # ====== filter out unsupport types ====== #
    if name is not None:
        name = as_tuple(name, t=string_types)
        alltensors = [t for t in alltensors
        if any((n == t.name.split('/')[-1] or
                n + ':0' == t.name.split('/')[-1]) for n in name)]
    if full_name is not None:
        full_name = as_tuple(full_name, t=string_types)
        alltensors = [t for t in alltensors
                      if any((n == t.name or
                              n + ':0' == t.name) for n in full_name)]
    return alltensors


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
    """

    def __init__(self, inputs, outputs, updates=[], defaults={}):
        # ====== validate input ====== #
        if isinstance(inputs, dict):
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
        self.return_list = return_list
        # ====== validate updates ====== #
        if isinstance(updates, dict):
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
                updates_ops = tf.group(*updates_ops)
            else: # already an tensorflow Ops
                updates_ops = updates
            # annotated updates from Graph
            updates_graph = ComputationGraph(outputs).updates
            if len(updates_graph) > 0:
                updates_graph = tf.group(*[tf.assign(v, new_v)
                    for v, new_v in updates_graph.iteritems()])
                updates_ops = tf.group(*[updates_ops, updates_graph])
            # merged updated
            self.updates_ops = updates_ops

    def __call__(self, *inputs, **kwargs):
        # dictionary as inputs
        if len(kwargs) == len(self.inputs_name):
            inputs = [kwargs[i] for i in self.inputs_name]
        # ====== create feed_dict ====== #
        feed_dict = {}
        for tensor, value in zip(self.inputs, inputs):
            feed_dict[tensor] = value
        feed_dict.update(self.defaults)
        # ====== run the output ====== #
        session = get_session()
        updated = session.run(self.outputs + [self.updates_ops],
                              feed_dict=feed_dict)
        # ====== get the results ====== #
        outputs = updated[:len(self.outputs)]
        if not self.return_list:
            outputs = outputs[0]
        return outputs


def function(inputs, outputs, updates=[], defaults={}):
    # ====== check inputs ====== #
    if inputs is None or len(as_tuple(inputs)) == 0:
        inputs = ComputationGraph(outputs).inputs
        print("[WARNING] inputs haven't specified, auto-inferred from Graph of "
              "outputs, graph inputs: %s" % ', '.join([str(i) for i in inputs]))
    return Function(inputs=inputs, outputs=outputs,
                    updates=updates, defaults=defaults)


# ===========================================================================
# Computational graph
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

    def __init__(self, outputs=None):
        outputs = flatten_list(as_list(outputs), level=None)
        self.outputs = [o for o in outputs if o is not None]
        self._get_variables()

    def _get_variables(self):
        """ Collect variables, updates and auxiliary variables.

        In addition collects all :class:`.Scan` ops and recurses in the
        respective inner Theano graphs. """
        _travelled_up = [] # to prevent recursive ops
        _travelled_down = [] # to prevent recursive ops

        def get_all_tensor_trace_down(x):
            """ recursively travel down the inputs tree to get all
            variables """
            variables = []
            op = x.op
            # ====== check travelled ops ====== #
            if op in _travelled_down:
                return variables
            else:
                _travelled_down.append(op)
            # ====== get all variable ====== #
            inputs = op._inputs
            variables += inputs
            for i in inputs:
                variables += get_all_tensor_trace_down(i)
            return variables

        def get_all_tensor_trace_up(x):
            """ travel up the outputs tree to get all variables"""
            variables = []
            # ====== check travelled ops ====== #
            for op in x.consumers():
                if op in _travelled_up:
                    continue
                else:
                    _travelled_up.append(op)
                # ====== get all variable ====== #
                inputs = [i for i in op._inputs if i != x]
                outputs = op._outputs
                variables += inputs + outputs
                for o in outputs:
                    variables += get_all_tensor_trace_up(o)
            return variables

        def create_tensor_iter(outputs):
            # if specific outputs is given
            if len(outputs) > 0:
                for o in outputs:
                    # travese each node of graph
                    all_variables = get_all_tensor_trace_down(o) + \
                        get_all_tensor_trace_up(o)
                    for v in all_variables:
                        yield v
            # get all variables and tensor within the graph
            else:
                graph = get_session().graph
                all_ops = graph.get_operations()
                for o in all_ops:
                    for v in o._inputs + o._outputs + o._control_inputs:
                        yield v
        # store all the updates embedded into the Tensor Variables
        updates = OrderedDict()
        shared_outputs = [o for o in self.outputs if is_variable(o)]
        usual_outputs = [o for o in self.outputs if not is_variable(o)]
        variables = shared_outputs
        trainable_variables = list(shared_outputs)
        inputs = []
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
        for v in create_tensor_iter(usual_outputs):
            if v.name in global_vars:
                variables.append(global_vars[v.name])
            if is_tensor(v):
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
        trainable_variables = [v for v in variables if is_variable(v)]
        # ====== get all updates and auxiliary variables ====== #
        for v in inputs + variables:
            if hasattr(v, 'tag'):
                # updates
                _ = getattr(v.tag, 'updates', OrderedDict())
                _ = OrderedDict([(i, j) for i, j in _.iteritems()
                                 if is_tensor(i)])
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
    def placeholders(self):
        """Inputs to the graph, excluding constants and shared variables."""
        return self._inputs

    @property
    def tensors(self):
        return [var for var in self.variables if
                var not in self.placeholders and
                var not in self.outputs]

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

    def get_roles(self, roles, match_all=False, exact=False):
        """ Return all variables with given roles """
        return [v for v in self.variables
                if has_roles(v, roles, match_all=match_all, exact=exact)]

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
