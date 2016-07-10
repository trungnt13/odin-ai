"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division
import logging
import warnings
import numbers
import cPickle
from contextlib import contextmanager
from collections import OrderedDict
from itertools import chain
from toolz import unique

import numpy as np

import theano
from theano import Variable
from theano import tensor as T
from theano.gof import graph
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.scan_module.scan_op import Scan
from theano.gof.graph import Constant
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable

from odin.roles import (add_role, has_roles, TRAINING, DEPLOYING,
                        AUXILIARY, PARAMETER, COLLECTED, COLLECTOR)
from odin.utils.decorators import singleton
from odin.utils import dict_union, as_shape_tuple
from odin.config import autoconfig, device

FLOATX = autoconfig.floatX
NPROCESSORS = device['n']
logger = logging.getLogger(__name__)


# ===========================================================================
# Shape helpers
# ===========================================================================
def _check_target(target):
    if autoconfig['device'] == 'cpu':
        target = None
    else:
        if target is None:
            target = 'dev0'
        elif isinstance(target, numbers.Number):
            target = 'dev%d' % (int(target) % NPROCESSORS)
        else:
            target = str(target)
    return target


def _auto_infer_shape(ops, *var, **kwargs):
    inputs = []
    for i in var:
        input_shape = (0 if s is None or s < 0 else s for s in get_shape(i))
        inputs.append(T.alloc(0, *input_shape))
    output_shape = ops(*inputs, **kwargs).shape.eval()
    return tuple(s if s else None for s in output_shape)


def add_shape(var, shape):
    # do nothing if not Number of tuple, list
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if not isinstance(shape, numbers.Number) and \
    not isinstance(shape, (tuple, list)):
        return

    shape = as_shape_tuple(shape)
    if len(shape) != var.ndim:
        raise ValueError('Variable has ndim={} but given shape has ndim={}'
                         '.'.format(var.ndim, len(shape)))
    # ====== NO override ====== #
    if hasattr(var.tag, 'shape'):
        raise Exception('Shape has already been added to given variable.')
    # ====== override or assign ====== #
    var.tag.shape = shape


def get_shape(x):
    """Return the shape of a tensor, this function search for predefined shape
    of `x` first, otherwise, return the theano shape

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).

    Parameters
    ----------
    none : bool
        allow None value in the shape tuple, otherwise, all None (and -1)
        dimensions are converted to intermediate shape variable
    """
    if not hasattr(x, 'shape'):
        raise Exception("Variable doesn't has shape attribute.")
    shape = x.shape
    if hasattr(x.tag, 'shape'):
        shape = x.tag.shape
    return shape


def ndim(x):
    return x.ndim


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
        if is_placeholder(i) and has_roles(i, TRAINING, exact=True):
            return True
    return False


# ===========================================================================
# VARIABLE MANIPULATION
# ===========================================================================
_CURRENT_VARIABLE_SCOPE = ""
_CREATED_VARIABLE = {}
# var id start from 0 and increasing to make sure no duplicate variable
_VAR_ID = 0


@contextmanager
def variable_scope(scope):
    global _CURRENT_VARIABLE_SCOPE
    old_scope = _CURRENT_VARIABLE_SCOPE
    _CURRENT_VARIABLE_SCOPE = str(scope)
    yield None
    _CURRENT_VARIABLE_SCOPE = old_scope


def variable(value, dtype=FLOATX, name=None, target=None):
    """Instantiate a tensor variable.
    """
    if name is None:
        global _VAR_ID
        name = 'VAR_%d' % _VAR_ID
        _VAR_ID += 1
    if len(_CURRENT_VARIABLE_SCOPE) > 0: # not global scope
        name = _CURRENT_VARIABLE_SCOPE + "/" + name

    # ====== validate inputs ====== #
    value = np.asarray(value, dtype=dtype)
    target = _check_target(target)

    kwargs = {}
    if target is not None:
        kwargs['target'] = target

    variable = theano.shared(value=value, name=name, strict=False, **kwargs)
    add_shape(variable, tuple(variable.shape.eval()))
    # ====== save all created variable ====== #
    if name in _CREATED_VARIABLE:
        raise Exception('Variable with the same name "%s" has been created '
                        'before.' % name)
    _CREATED_VARIABLE[name] = variable # save original shared variables
    return variable


def placeholder(shape, dtype=FLOATX, name=None, for_training=False):
    """Instantiate an input data placeholder variable.
    """
    shape = as_shape_tuple(shape)
    broadcast = tuple([True if i == 1 else False for i in shape])
    # ====== Modify add name prefix ====== #
    placeholder = T.TensorType(dtype, broadcast)(name)
    if for_training:
        add_role(placeholder, TRAINING)
    else:
        add_role(placeholder, DEPLOYING)
    # store the predefined shape of placeholder
    add_shape(placeholder, shape)
    return placeholder


def constant(value, dtype=None, shape=None, name='Const'):
    x = T.constant(value, dtype=dtype,
                   ndim=None if shape is None else len(shape),
                   name=name)
    add_shape(x, eval(x.shape))
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
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = list(outputs)
        self._get_variables()

    def _get_variables(self):
        """Collect variables, updates and auxiliary variables.

        In addition collects all :class:`.Scan` ops and recurses in the
        respective inner Theano graphs.

        """
        updates = OrderedDict()

        shared_outputs = [o for o in self.outputs if is_trainable_variable(o)]
        usual_outputs = [o for o in self.outputs if not is_trainable_variable(o)]
        variables = shared_outputs

        if usual_outputs:
            # Sort apply nodes topologically, get variables and remove
            # duplicates
            inputs = graph.inputs(self.outputs)
            sorted_apply_nodes = graph.io_toposort(inputs, usual_outputs)
            self.scans = list(unique([node.op for node in sorted_apply_nodes
                                     if isinstance(node.op, Scan)],
                                     key=lambda op: id(op)))
            self._scan_graphs = [ComputationGraph(scan.outputs)
                                 for scan in self.scans]

            seen = set()
            main_vars = (
                [var for var in list(chain(
                    *[apply_node.inputs for apply_node in sorted_apply_nodes]))
                 if not (var in seen or seen.add(var))] +
                [var for var in self.outputs if var not in seen])

            # While preserving order add auxiliary variables, and collect
            # updates
            seen = set()
            # Intermediate variables could be auxiliary
            seen_avs = set(main_vars)
            variables = []
            for var in main_vars:
                variables.append(var)
                # updates
                updates = dict_union(updates,
                                     getattr(var.tag, 'updates', OrderedDict()))
                # auxiliary_variables
                for _ in getattr(var.tag, 'auxiliary_variables', []):
                    if _ not in seen and \
                    not (_ in seen_avs or seen_avs.add(_)):
                        variables.append(_)

        # If trainable_variables is assigned default_update (cloned), we cannot eval()
        # it to get the real numpy array value, hence, try to trace back
        # original shared variable
        def shared_variable_filter(var):
            if is_trainable_variable(var) and hasattr(var, 'default_update'):
                for v in _CREATED_VARIABLE:
                    if v.name == var.name and v.ndim == var.ndim:
                        return v
            return var
        self.variables = map(shared_variable_filter, variables)
        self.updates = updates

    # ==================== Get variables ==================== #
    @property
    def placeholders(self):
        """Inputs to the graph, excluding constants and shared variables."""
        return [var for var in self.variables if is_placeholder(var)]

    @property
    def intermediary_variables(self):
        return [var for var in self.variables if
                var not in self.placeholders and
                var not in self.outputs]

    @property
    def trainable_variables(self):
        return [var for var in self.variables if is_trainable_variable(var)]

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
