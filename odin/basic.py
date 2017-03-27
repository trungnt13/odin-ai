from __future__ import print_function, division, absolute_import

import re
import numbers
import warnings
from collections import OrderedDict

import numpy as np

from odin.utils import as_shape_tuple, struct
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
class VariableRole(object):
    """Base class for all variable roles."""

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __repr__(self):
        return re.sub(r'(?!^)([A-Z]+)', r'_\1',
                      self.__class__.__name__[:-4]).upper()


class AuxiliaryRole(VariableRole):
    pass
#: Variables added to the graph as annotations
AUXILIARY = AuxiliaryRole()


# ==================== Variational ==================== #
class VariationalRole(VariableRole):
    pass
#: Variational statistics
VARIATIONAL = VariationalRole()


class VariationalMean(VariationalRole):
    pass
VARIATIONAL_MEAN = VariationalMean()


class VariationalLogsigma(VariationalRole):
    pass
VARIATIONAL_LOGSIGMA = VariationalLogsigma()


# ==================== Role for Variable ==================== #
class CostRole(VariableRole):
    pass
#: A scalar cost that can be used to train or regularize
COST = CostRole()


class ParameterRole(VariableRole):
    pass
#: A parameter of the model
PARAMETER = ParameterRole()


class ActivationParameterRole(ParameterRole):
    pass
#: A parameter of the model
ACTIVATION_PARAMETER = ActivationParameterRole()


class WeightRole(ParameterRole):
    pass
#: The weight matrices of linear transformations
WEIGHT = WeightRole()


class BiasRole(ParameterRole):
    pass
#: Biases of linear transformations
BIAS = BiasRole()


class InitialStateRole(ParameterRole):
    pass


#: Initial state of a recurrent network
INITIAL_STATE = InitialStateRole()


class FilterRole(WeightRole):
    pass
#: The filters (kernels) of a convolution operation
FILTER = FilterRole()


class DropoutRole(VariableRole):
    pass
#: Inputs with applied dropout
DROPOUT = DropoutRole()


# ==================== Optimizer Algorithm roles ==================== #
class OptimizerHyperParameterRole(VariableRole):
    pass
#: Shared variables used in algorithms updates
OPTIMIZER_HYPER_PARAMETER = OptimizerHyperParameterRole()


class LearningRateRole(OptimizerHyperParameterRole):
    pass
LEARNING_RATE = LearningRateRole()


# ==================== Embedding ==================== #
class EmbeddingWeights(WeightRole):
    pass
#: weights for embedding operator
EMBEDDING = EmbeddingWeights()


# ==================== Batch normalization roles ==================== #
class BatchNormPopulationStatisticsRole(VariableRole):
    pass
#: base role for batch normalization population statistics
BATCH_NORM_POPULATION_STATISTICS = BatchNormPopulationStatisticsRole()


class BatchNormPopulationMeanRole(BatchNormPopulationStatisticsRole):
    pass
#: mean activations accumulated over the dataset
BATCH_NORM_POPULATION_MEAN = BatchNormPopulationMeanRole()


class BatchNormPopulationInvStdRole(BatchNormPopulationStatisticsRole):
    pass
#: standard deviations of activations accumulated over the dataset
BATCH_NORM_POPULATION_INVSTD = BatchNormPopulationInvStdRole()


class BatchNormScaleParameterRole(ParameterRole):
    pass
#: role given to the scale parameter, referred to as "scale" (or "gamma") in the
# batch normalization manuscript, applied after normalizing.
BATCH_NORM_SCALE_PARAMETER = BatchNormScaleParameterRole()


class BatchNormShiftParameterRole(BiasRole):
    pass
#: role given to the shift parameter, referred to as "beta" in the
# batch normalization manuscript, applied after normalizing and scaling.
# Inherits from BIAS, because there really is no functional difference
# with a normal bias, and indeed these are the only biases present
# inside a BatchNormalizedMLP.
BATCH_NORM_SHIFT_PARAMETER = BatchNormShiftParameterRole()


# ===========================================================================
# Variable tagging
# ===========================================================================
def _check_tag(var):
    if not hasattr(var, 'tag'):
        var.tag = struct()


# ==================== shape ==================== #
def add_shape(var, shape):
    _check_tag(var)
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


# ==================== updates ==================== #
def add_updates(var, key, value):
    r""" Annotate updates to a given var, hence, this updates will
    be used when create function

    """
    _check_tag(var)
    updates = getattr(var.tag, 'updates', OrderedDict())
    updates[key] = value
    var.tag.updates = updates


def add_role(var, role):
    r"""Add a role to a given Theano variable.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        The variable to assign the new role to.
    role : :class:`.VariableRole` instance

    Notes
    -----
    Some roles are subroles of others (e.g. :const:`WEIGHT` is a subrole
    of :const:`PARAMETER`). This function will not add a role if a more
    specific role has already been added. If you need to replace a role
    with a parent role (e.g. replace :const:`WEIGHT` with
    :const:`PARAMETER`) you must do so manually.

    Examples
    --------
    >>> from theano import tensor
    >>> W = tensor.matrix()
    >>> from blocks.roles import PARAMETER, WEIGHT
    >>> add_role(W, PARAMETER)
    >>> print(*W.tag.roles)
    PARAMETER
    >>> add_role(W, WEIGHT)
    >>> print(*W.tag.roles)
    WEIGHT
    >>> add_role(W, PARAMETER)
    >>> print(*W.tag.roles)
    WEIGHT

    """
    _check_tag(var)
    roles = getattr(var.tag, 'roles', [])
    roles = [old_role for old_role in roles
             if not isinstance(role, old_role.__class__)]
    # add a role if it isn't in the list
    if not any(isinstance(old_role, role.__class__) for old_role in roles):
        roles += [role]
    var.tag.roles = roles


def add_auxiliary_variable(var, auxiliary, roles=None):
    r""" Annotate auxiliary variable to a given var

    """
    _check_tag(var)
    auxiliary_variables = getattr(var.tag, 'auxiliary_variables', [])
    add_role(auxiliary, AUXILIARY)
    if roles is not None:
        for role in roles:
            add_role(auxiliary, role)
    auxiliary_variables.append(auxiliary)
    var.tag.auxiliary_variables = list(set(auxiliary_variables))


def has_roles(var, roles, match_all=False, exact=False):
    r"""Test if a variable has given roles taking subroles into account.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        Variable being queried.
    roles : an iterable of :class:`.VariableRole` instances.
    match_all : bool, optional
        If ``True``, checks if the variable has all given roles.
        If ``False``, any of the roles is sufficient.
        ``False`` by default.
    exact : bool, optional
        If ``True``, use ``==`` for comparison to get exactly same roles.
        If ``False``, use isinstance for comparison, hence, also match the
        decesdant roles.

    """
    # don't have tag attribute
    if not hasattr(var, 'tag'):
        return False
    # prepare roles
    if not hasattr(roles, '__iter__'):
        roles = [roles]
    var_roles = getattr(var.tag, 'roles', [])
    if not exact:
        matches = (any(isinstance(var_role, role.__class__) for
                       var_role in var_roles) for role in roles)
    else:
        matches = (any(var_role.__class__ == role.__class__ for
                       var_role in var_roles) for role in roles)
    return all(matches) if match_all else any(matches)


def get_roles(var):
    return getattr(var.tag, 'roles', [])
