"""Annotated computation graph management."""
from __future__ import print_function, absolute_import, division
import logging
import warnings
import numbers
import cPickle
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

from odin.roles import (add_role, has_roles,
                        AUXILIARY, PARAMETER, COLLECTED, COLLECTOR,
                        TRAINING)

from odin.utils import dict_union, as_shape_tuple
from odin.config import autoconfig, device
# from .annotations import add_annotation, Annotation  # noqa

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
# ComputationGraph
# ===========================================================================
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
    shared_variables : list of :class:`~tensor.TensorSharedVariable`
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
        self._has_inputs = {}

    def __iter__(self):
        return iter(self.variables)

    @property
    def inputs(self):
        """Inputs to the graph, excluding constants and shared variables."""
        return [var for var in self.variables if is_placeholder(var)]

    @property
    def intermediary_variables(self):
        return [var for var in self.variables if
                var not in self.inputs and
                var not in self.outputs]

    @property
    def shared_variables(self):
        return [var for var in self.variables if is_trainable_variable(var)]

    @property
    def parameters(self):
        return [var for var in self.shared_variables
                if has_roles(var, [PARAMETER])]

    @property
    def auxiliary_variables(self):
        return [var for var in self.variables if has_roles(var, [AUXILIARY])]

    @property
    def scan_variables(self):
        """Variables of Scan ops."""
        return list(chain(*[g.variables for g in self._scan_graphs]))

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
                for annotation in getattr(var.tag, 'annotations', []):
                    if annotation not in seen:
                        seen.add(annotation)
                        new_avs = [
                            av for av in annotation.auxiliary_variables
                            if not (av in seen_avs or seen_avs.add(av))]
                        variables.extend(new_avs)
                        updates = dict_union(updates, annotation.updates)

        # If shared_variables is assigned default_update (cloned), we cannot eval()
        # it to get the real numpy array value, hence, try to trace back
        # original shared variable
        def shared_variable_filter(var):
            if is_trainable_variable(var) and hasattr(var, 'default_update'):
                for annotation in var.tag.annotations:
                    if hasattr(annotation, var.name) and \
                       is_trainable_variable(getattr(annotation, var.name)):
                        return getattr(annotation, var.name)
            return var
        self.variables = map(shared_variable_filter, variables)
        self.updates = updates

    def dict_of_inputs(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.inputs}

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        Parameters
        ----------
        replacements : dict
            The mapping from variables to be replaced to the corresponding
            substitutes.

        Examples
        --------
        >>> import theano
        >>> from theano import tensor, function
        >>> x = tensor.scalar('x')
        >>> y = x + 2
        >>> z = y + 3
        >>> a = z + 5

        Let's suppose we have dependent replacements like

        >>> replacements = {y: x * 2, z: y * 3}
        >>> cg = ComputationGraph([a])
        >>> theano.pprint(a)  # doctest: +NORMALIZE_WHITESPACE
        '(((x + TensorConstant{2}) + TensorConstant{3}) +
        TensorConstant{5})'
        >>> cg_new = cg.replace(replacements)
        >>> theano.pprint(
        ...     cg_new.outputs[0])  # doctest: +NORMALIZE_WHITESPACE
        '(((x * TensorConstant{2}) * TensorConstant{3}) +
        TensorConstant{5})'

        First two sums turned into multiplications

        >>> float(function(cg_new.inputs, cg_new.outputs)(3.)[0])
        23.0

        """
        # Due to theano specifics we have to make one replacement in time
        replacements = OrderedDict(replacements)

        outputs_cur = self.outputs

        # `replacements` with previous replacements applied. We have to track
        # variables in the new graph corresponding to original replacements.
        replacement_keys_cur = []
        replacement_vals_cur = []
        # Sort `replacements` in topological order
        # variables in self.variables are in topological order
        remaining_replacements = replacements.copy()
        for variable in self.variables:
            if variable in replacements:
                if has_roles(variable, [AUXILIARY]):
                    warnings.warn(
                        "replace method was asked to replace a variable ({}) "
                        "that is an auxiliary variable.".format(variable))
                replacement_keys_cur.append(variable)
                # self.variables should not contain duplicates,
                # otherwise pop() may fail.
                replacement_vals_cur.append(
                    remaining_replacements.pop(variable))

        # if remaining_replacements is not empty
        if remaining_replacements:
            warnings.warn(
                "replace method was asked to replace a variable(s) ({}) "
                "that is not a part of the computational "
                "graph.".format(str(remaining_replacements.keys())))

        # Replace step-by-step in topological order
        while replacement_keys_cur:
            replace_what = replacement_keys_cur[0]
            replace_by = replacement_vals_cur[0]
            # We also want to make changes in future replacements
            outputs_new = theano.clone(
                outputs_cur + replacement_keys_cur[1:] +
                replacement_vals_cur[1:],
                replace={replace_what: replace_by})
            # Reconstruct outputs, keys, and values
            outputs_cur = outputs_new[:len(outputs_cur)]
            replacement_keys_cur = outputs_new[len(outputs_cur):
                                               len(outputs_cur) +
                                               len(replacement_keys_cur) - 1]
            replacement_vals_cur = outputs_new[len(outputs_cur) +
                                               len(replacement_keys_cur):]

        return ComputationGraph(outputs_cur)

    def get_theano_function(self, additional_updates=None, **kwargs):
        r"""Create Theano function from the graph contained.

        Parameters
        ----------
        \*\*kwargs : dict
            Keyword arguments to theano.function.
            Useful for specifying compilation modes or profiling.

        """
        updates = self.updates
        if additional_updates:
            updates = dict_union(updates, OrderedDict(additional_updates))
        return theano.function(self.inputs, self.outputs, updates=updates,
                               **kwargs)

    # def get_snapshot(self, data):
    #     """Evaluate all role-carrying Theano variables on given data.

    #     Parameters
    #     ----------
    #     data : dict of (data source, data) pairs
    #         Data for input variables. The sources should match with the
    #         names of the input variables.

    #     Returns
    #     -------
    #     Dictionary of (variable, variable value on given data) pairs.

    #     """
    #     role_variables = [var for var in self.variables
    #                       if hasattr(var.tag, "roles") and
    #                       not is_trainable_variable(var)]
    #     value_holders = [shared_like(var) for var in role_variables]
    #     function = self.get_theano_function(equizip(value_holders,
    #                                                 role_variables))
    #     function(*(data[input_.name] for input_ in self.inputs))
    #     return OrderedDict([(var, value_holder.get_value(borrow=True))
    #                         for var, value_holder in equizip(role_variables,
    #                                                          value_holders)])

    def has_inputs(self, variable):
        """Check if a variable depends on input variables.

        Returns
        -------
        bool
            ``True`` if the given variable depends on input variables,
            ``False`` otherwise.

        """
        if variable not in self._has_inputs:
            self._has_inputs[variable] = False
            if is_placeholder(variable):
                self._has_inputs[variable] = True
            elif getattr(variable, 'owner', None):
                for dependancy in variable.owner.inputs:
                    if self.has_inputs(dependancy):
                        self._has_inputs[variable] = True
        return self._has_inputs[variable]
