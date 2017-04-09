from __future__ import print_function, division, absolute_import

import re
import inspect
import numbers
import warnings
from decorator import decorator
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np

from odin.utils import struct, is_number, as_tuple, flatten_list
from odin.config import get_backend


# ===========================================================================
# TRaining flag
# ===========================================================================
__IS_TRAINING = False


def is_training():
    if __IS_TRAINING:
        return 1
    return 0


def set_training(train):
    global __IS_TRAINING
    __IS_TRAINING = train


# ===========================================================================
# Variable ROles
# ===========================================================================
class Role(object):
    """Base class for all roles."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("This is class is only for annotation, you cannot "
                           "create instance from this class.")


class Randomization(Role):
    """Base class for all variable roles."""
    pass


class Variable(Role):
    """Base class for all variable roles."""
    pass


class Auxiliary(Variable):
    """ Variables added to the graph as annotations """
    pass


class GradientsNorm(Auxiliary):
    pass


class AccuracyValue(Auxiliary):
    pass


class ConfusionMatrix(Auxiliary):
    pass


class EarlyStop(Auxiliary):
    pass


# ==================== Role for Cost and Objective ==================== #
class ObjectiveCost(Variable):
    pass


class TrainingCost(ObjectiveCost):
    pass


class RegularizeCost(ObjectiveCost):
    pass


# ==================== Variational ==================== #
class Variational(Variable):
    """ All role related to variational inference """
    pass


class VariationalMean(Variational):
    pass


class VariationalLogsigma(Variational):
    pass


# ==================== Role for Trainable Variable ==================== #
class Parameter(Variable):
    pass


class ActivationParameter(Parameter):
    pass


class Weight(Parameter):
    pass


class Bias(Parameter):
    pass


class InitialState(Parameter):
    """ Initial state of a recurrent network """
    pass


class ConvKernel(Weight):
    """ The filters (kernels) of a convolution operation """
    pass


class Dropout(Variable):
    """ Inputs with applied dropout """
    pass


# ==================== Optimizer Algorithm roles ==================== #
class OptimizerHyperParameter(Variable):
    """ Shared variables used in algorithms updates """
    pass


class LearningRate(OptimizerHyperParameter):
    pass


class LearningRateDecay(OptimizerHyperParameter):
    pass


# ==================== Embedding ==================== #
class EmbeddingWeight(Weight):
    """ weights for embedding operator """
    pass


# ==================== Batch normalization roles ==================== #
class BatchNorm(Variable):
    """ base role for batch normalization population statistics """
    pass


class BatchNormPopulationMean(BatchNorm):
    """ mean activations accumulated over the dataset """
    pass


class BatchNormPopulationInvStd(BatchNorm):
    """ standard deviations of activations accumulated over the dataset """
    pass


class BatchNormScaleParameter(Parameter, BatchNorm):
    """ role given to the scale parameter, referred to as "scale" (or "gamma") in the """
    pass


class BatchNormShiftParameter(Bias, BatchNorm):
    """ role given to the shift parameter, referred to as "beta" in the
    batch normalization manuscript, applied after normalizing and scaling.
    Inherits from BIAS, because there really is no functional difference
    with a normal bias, and indeed these are the only biases present
    inside a BatchNormalizedMLP.
    """
    pass


# ===========================================================================
# Variable tagging
# ===========================================================================
def _check_tag(var):
    if not hasattr(var, 'tag'):
        tag = struct()
        tag.roles = []
        var.tag = tag
    return var


# ===========================================================================
# Basic shape helper
# ===========================================================================
def _is_tensor(x):
    if get_backend() == 'tensorflow':
        import tensorflow as tf
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return True
    elif get_backend() == 'theano':
        from theano.gof.graph import Constant
        from theano.tensor.sharedvar import SharedVariable
        from theano import Variable
        if isinstance(x, (Constant, Variable, SharedVariable)):
            return True
    return False


def as_shape_tuple(shape):
    if is_number(shape):
        shape = (int(shape),)
    elif _is_tensor(shape):
        shape = (shape,)
    else:
        if not isinstance(shape, (tuple, list, np.ndarray)):
            raise ValueError('We only accept shape in tuple, list or numpy.ndarray.')
        shape = tuple([(int(i) if i > 0 else None) if is_number(i) else i
                       for i in shape])
        if len([i for i in shape if i is None]) >= 2:
            raise Exception('Shape tuple can only have 1 unknown dimension.')
    return shape


def add_shape(var, shape):
    try:
        _check_tag(var)
    except:
        return var
    # do nothing if not Number of tuple, list
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    # not Number or None, not a valid shape
    if any(not isinstance(s, numbers.Number) and s is not None
           for s in shape):
        return
    # check shape tuple
    try:
        shape = as_shape_tuple(shape)
    except Exception as e:
        print("Cannot process shape=%s, exception:%s" % (str(shape), str(e)))
        return var
    # check ndim
    ndim = var.ndim if hasattr(var, 'ndim') else var.get_shape().ndims
    if len(shape) != ndim:
        raise ValueError('Variable has ndim={} but given shape has ndim={}'
                         '.'.format(ndim, len(shape)))
    # ====== NO override ====== #
    if hasattr(var.tag, 'shape') and var.tag.shape != shape:
        warnings.warn('Variable already had shape=%s, and the given shape is: %s'
                      '.' % (var.tag.shape, shape))
    # ====== override or assign ====== #
    else:
        var.tag.shape = shape
    return var


def get_shape(x, native=False):
    """Return the shape of a tensor, this function search for predefined shape
    of `x` first, otherwise, return the theano shape

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).

    Parameters
    ----------
    x: theano or tensorflow variable, numpy.ndarray
        variable for getting the shape
    not_none : bool
        if `not_none`=True, does not allow None in returned shape tuple.
        Default value is False
    native : bool
        if True, return the native shape information returned by backend (i.e.
        object shape not int shape)
    """
    # ====== get default shape ====== #
    if hasattr(x, 'tag') and hasattr(x.tag, 'shape'):
        shape = x.tag.shape
        # just ensure the tagged shape equal to actual tensorflow shape
        if get_backend() == 'tensorflow':
            tensorflow_shape = x.get_shape().as_list()
            if len(shape) != len(tensorflow_shape) or \
            any(i != j for i, j in zip(shape, tensorflow_shape)
                    if i is not None and j is not None):
                raise ValueError("The tagged shape is %s, but the system shape is "
                                "%s." % (shape, x.get_shape().as_list()))
    elif hasattr(x, 'get_shape'):
        shape = tuple(x.get_shape().as_list())
    elif hasattr(x, 'shape'):
        shape = x.shape
    else:
        raise ValueError('Cannot get_shape of variable: ' + str(x))
    # ====== check tag shape ====== #
    if native and isinstance(shape, (tuple, list)):
        if get_backend() == 'theano':
            native_shape = x.shape
        else:
            import tensorflow as tf
            native_shape = tf.shape(x)
        # return a mix of native tensor variable shape, and int shape
        return tuple([native_shape[i] if j is None or j < 0 else j
                      for i, j in enumerate(shape)])
    return shape


# ===========================================================================
# Basic Role helper
# ===========================================================================
def add_updates(var, key, value):
    r""" Annotate updates to a given var, hence, this updates will
    be used when create function

    Note
    ----
    updates won't be serialized during pickling of any variables.

    """
    try:
        _check_tag(var)
    except:
        return var
    updates = getattr(var.tag, 'updates', OrderedDict())
    updates[key] = value
    var.tag.updates = updates


def add_role(variables, roles=None):
    r"""Add a role to a given variable.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        The variable to assign the new role to.
    roles : :subclass:`Role`
        this roles will be concatenated with current roles scope.

    Notes
    -----
    Some roles are subroles of others (e.g. :class:`Weight` is a subrole
    of :class:`Parameter`). This function will not add a role if a more
    specific role has already been added. If you need to replace a role
    with a parent role (e.g. replace :class:`Weight` with
    :class:`Parameter`) you must do so manually.

    """
    # create tag attribute for variable
    for var in as_tuple(variables):
        try:
            _check_tag(var)
        except:
            continue
        roles = [r for r in as_tuple(roles)
                 if isinstance(r, type) and issubclass(r, Role)]
        # append roles scope
        roles += get_current_role_scope()
        var_roles = list(getattr(var.tag, 'roles', []))
        var_roles = var_roles + roles
        # ====== shrink the roles so there is NO subrole ====== #
        var_roles = [r for r in var_roles
                     if not any(r != r0 and issubclass(r0, r) for r0 in var_roles)]
        # ====== adding new role ====== #
        var.tag.roles = var_roles
    return variables


def add_auxiliary_variable(var, auxiliary, roles=None):
    r""" Annotate auxiliary variable to a given var

    """
    try:
        _check_tag(var)
    except:
        return var
    auxiliary_variables = getattr(var.tag, 'auxiliary_variables', [])
    add_role(auxiliary, Auxiliary)
    if roles is not None:
        for r in roles:
            add_role(auxiliary, r)
    auxiliary_variables.append(auxiliary)
    var.tag.auxiliary_variables = list(set(auxiliary_variables))
    return var


def has_roles(var, roles, match_all=False, exact=False):
    r"""Test if a variable has given roles taking subroles into account.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        Variable being queried.
    roles : an iterable of :subclass:`.Role`.
    match_all : bool, optional
        If ``True``, checks if the variable has all given roles.
        If ``False``, any of the roles is sufficient.
        ``False`` by default.
    exact : bool, optional
        If ``True``, use ``==`` for comparison to get exactly same roles.
        If ``False``, use issubclass for comparison, hence, also match the
        decesdant roles.

    """
    # don't have tag attribute
    if not hasattr(var, 'tag'):
        return False
    # prepare roles
    roles = [r for r in as_tuple(roles) if issubclass(r, Role)]
    var_roles = getattr(var.tag, 'roles', [])
    if not exact:
        matches = (any(issubclass(var_role, match_role) for var_role in var_roles)
                   for match_role in roles)
    else:
        matches = (any(var_role == match_role for var_role in var_roles)
                   for match_role in roles)
    return all(matches) if match_all else any(matches)


def get_roles(var):
    return getattr(var.tag, 'roles', [])


# ===========================================================================
# Role context manager
# ===========================================================================
__ROLE_STACK = [[]]


def get_current_role_scope():
    return __ROLE_STACK[-1]


def output_roles(roles=None):
    """ A decorators to assign specific role to all outputs of a function.

    Example
    -------
    >>> with role_scope(Variational):
    ...     @output_roles(Weight)
    ...     def func():
    ...         return K.variable(np.random.rand(12, 8))
    ...     X = func()
    >>> print(X.tag.roles)
    ... # [<class 'odin.basic.Weight'>, <class 'odin.basic.Variational'>]
    """
    @decorator
    def add_role_to_outputs(func, *args, **kwargs):
        outputs = func(*args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            for o in outputs:
                add_role(o, roles)
        else:
            add_role(outputs, roles)
        return outputs

    # roles are not specified, given function directly
    if inspect.isfunction(roles) or inspect.ismethod(roles):
        func = roles
        roles = []
        return add_role_to_outputs(func)
    # roles are specified
    else:
        roles = [r for r in as_tuple(roles)
                 if isinstance(r, type) and issubclass(r, Role)]
    return add_role_to_outputs


@contextmanager
def role_scope(*roles):
    """
    Example
    -------
    >>> X = K.variable(np.random.rand(12, 8))
    >>> with role_scope(Weight, Variational, VariationalMean):
    ...     add_role(X)
    >>> print(X.tag.roles)
    ... # [<class 'odin.basic.Weight'>, <class 'odin.basic.VariationalMean'>]
    """
    roles = [r for r in flatten_list(roles, level=None)
             if isinstance(r, type) and issubclass(r, Role)]
    # ====== shrink the roles so there is NO subrole ====== #
    roles = __ROLE_STACK[-1] + roles
    roles = [r for r in roles
             if not any(r != r0 and issubclass(r0, r) for r0 in roles)]
    __ROLE_STACK.append(roles)
    yield roles
    __ROLE_STACK.pop()
