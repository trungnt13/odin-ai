from __future__ import print_function, division, absolute_import

import sys
import inspect
from six import string_types
from decorator import decorator
from contextlib import contextmanager

import tensorflow as tf

from odin.utils import as_tuple, flatten_list

# ===========================================================================
# Variable roles
# ===========================================================================
class Role(object):
  """Base class for all roles."""

  def __init__(self, *args, **kwargs):
    raise RuntimeError("This is class is only for annotation, you cannot "
                       "create instance from this class.")

# ==================== basic role tyupes ==================== #
class Randomization(Role):
  """Base class for all Randomizaton roles."""
  pass

class Variable(Role):
  """Base class for all variable roles."""
  pass

class variables(Variable):
  pass

# ==================== NNOp related ==================== #
class NNOpOutput(Role):
  """Base class for all Output of NNOp."""
  pass

# ==================== Optimizer Algorithm roles ==================== #
class OptimizerVariable(Variable):
  pass

class OptimizerHyperParameter(OptimizerVariable):
  """ Shared variables used in algorithms updates """
  pass

class LearningRate(OptimizerHyperParameter):
  pass

class LearningRateDecay(OptimizerHyperParameter):
  pass

class GraidentsClipping(OptimizerHyperParameter):
  pass

class GraidentsClippingNorm(GraidentsClipping):
  pass

class GraidentsClippingValue(GraidentsClipping):
  pass

# ******************** Role for Cost and Objective ******************** #
class Auxiliary(Variable):
  """ Variables added to the graph as annotations """
  pass

# DifferentialLoss
class DifferentialLoss(Auxiliary):
  pass

class RegularizeLoss(DifferentialLoss):
  pass

# DifferentialLoss
class MonitoringLoss(Auxiliary):
  pass

class GradientsNorm(MonitoringLoss):
  pass

class AccuracyValue(MonitoringLoss):
  pass

class ConfusionMatrix(MonitoringLoss):
  pass

class EarlyStop(MonitoringLoss):
  pass

# ******************** Role for Variational ******************** #
class Variational(Variable):
  """ All role related to variational inference """
  pass

class VariationalMean(Variational):
  pass

class VariationalLogsigma(Variational):
  pass

# ******************** Role for Parameters ******************** #
class Parameter(Variable):
  pass

# ====== anything trainable ====== #
class TrainableParameter(Parameter):
  pass

class trainable_variables(TrainableParameter):
  pass

# ====== others trainable ====== #
class ActivationParameter(Parameter):
  pass

class Weight(TrainableParameter):
  pass

class Bias(TrainableParameter):
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

class BatchNormScaleParameter(Weight, BatchNorm):
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
# Helpers
# ===========================================================================
_all_roles_name = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
  if inspect.isclass(obj) and issubclass(obj, Role):
    _all_roles_name[name] = obj

def name_to_roles(name):
  return _all_roles_name.get(name, name)

# ===========================================================================
# Basic Role helper
# ===========================================================================
def _add_to_collection_no_duplication(name, var):
  if var not in tf.get_collection(str(name)):
    tf.add_to_collection(name, var)

def add_roles(variables, roles):
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
  if roles is None:
    return variables
  roles = tuple([name_to_roles(r) for r in as_tuple(roles)])
  # create tag attribute for variable
  for var in as_tuple(variables):
    # append roles scope
    var_roles = get_roles(var, return_string=False) + \
        roles + \
        get_current_role_scope()
    # ====== handle string roles first ====== #
    _ = []
    for r in var_roles:
      if isinstance(r, string_types):
        _add_to_collection_no_duplication(r, var)
      elif isinstance(r, type) and issubclass(r, Role):
        _.append(r)
    var_roles = _
    # ====== shrink the roles so there is NO subrole ====== #
    new_roles = []
    for r in var_roles:
      # issubclass(r0=LearningRate, r=OptimizerVariable) = True
      # hence, remove var from `r` collection
      if any(r != r0 and issubclass(r0, r) for r0 in var_roles):
        r_collection = tf.get_collection_ref(r.__name__)
        if var in r_collection:
          r_collection.remove(var)
      else:
        new_roles.append(r)
    # ====== adding new role ====== #
    for r in new_roles:
      _add_to_collection_no_duplication(r.__name__, var)
  return variables

def _compare_role(var_role, match_role, exact):
  """ check if r1 is subclass of r2, or
  if r1 or r2 is string, r1 is equal r2

  Return
  ------
  True if matching
  False otherwise
  """
  # String types role
  if isinstance(var_role, string_types) or \
  isinstance(match_role, string_types):
    if inspect.isclass(var_role):
      var_role = var_role.__name__
    if inspect.isclass(match_role):
      match_role = match_role.__name__
    return var_role == match_role
  # subclass of Role
  return var_role == match_role if exact else \
  issubclass(var_role, match_role)

def has_roles(var, roles, match_all=False, exact=False):
  r"""Test if a variable has given roles taking subroles into account.

  Parameters
  ----------
  var : :class:`~tensor.TensorVariable`
      Variable being queried.
  roles : an iterable of :subclass:`.Role` or `str`
      List of all roles to match (role can come from `tf.GraphKeys`)
  match_all : bool, optional
      If ``True``, checks if the variable has all given roles.
      If ``False``, any of the roles is sufficient.
      ``False`` by default.
  exact : bool, optional
      If ``True``, use ``==`` for comparison to get exactly same roles.
      If ``False``, use issubclass for comparison, hence, also match the
      decesdant roles.

  """
  # prepare roles
  roles = [name_to_roles(r) if isinstance(r, string_types) else r
           for r in as_tuple(roles)
           if isinstance(r, string_types) or issubclass(r, Role)]
  var_roles = get_roles(var, return_string=False)
  matches = [any(_compare_role(var_role, match_role, exact)
             for var_role in var_roles)
             for match_role in roles]
  return all(matches) if match_all else any(matches)

def get_roles(var, return_string=True):
  """
  Parameters
  ----------
  var: `Tensor`
  return_string: bool
      if True, return the string which are name of all roles
      otherwise convert role to actual class and return them
  """
  roles = []
  for r, var_list in tf.get_default_graph()._collections.items():
    if var in var_list:
      if not return_string:
        roles.append(name_to_roles(r))
      else:
        roles.append(r)
  # always return the same order of roles
  return as_tuple(sorted(roles,
      key=lambda x: x if isinstance(x, string_types) else x.__name__))

# ===========================================================================
# Role context manager
# ===========================================================================
__ROLE_STACK = [[]]

def get_current_role_scope():
  return tuple(__ROLE_STACK[-1])

def return_roles(roles=None):
  """ A decorators to assign specific role to all outputs of a function.

  Example
  -------
  >>> with role_scope(Variational):
  ...     @return_roles(Weight)
  ...     def func():
  ...         return K.variable(np.random.rand(12, 8))
  ...     X = func()
  >>> print(X.tag.roles)
  ... # [<class 'odin.basic.Weight'>, <class 'odin.basic.Variational'>]
  """
  @decorator
  def addrole_to_outputs(func, *args, **kwargs):
    outputs = func(*args, **kwargs)
    if isinstance(outputs, (tuple, list)):
      for o in outputs:
        add_roles(o, roles)
    else:
      add_roles(outputs, roles)
    return outputs

  # roles are not specified, given function directly
  if inspect.isfunction(roles) or inspect.ismethod(roles):
    func = roles
    roles = []
    return addrole_to_outputs(func)
  # roles are specified
  else:
    roles = [r for r in as_tuple(roles)
             if isinstance(r, type) and issubclass(r, Role)]
  return addrole_to_outputs

@contextmanager
def role_scope(*roles):
  """
  Example
  -------
  >>> X = K.variable(np.random.rand(12, 8))
  >>> with role_scope(Weight, Variational, VariationalMean):
  ...     add_roles(X)
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
