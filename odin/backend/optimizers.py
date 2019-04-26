# ===========================================================================
# This module is created based on the code from libraries: Lasagne
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import numbers
from six import add_metaclass, string_types
from abc import ABCMeta, abstractproperty
from collections import OrderedDict, defaultdict

import numpy as np
import tensorflow as tf

from odin.autoconfig import CONFIG, get_session
from odin.utils import as_tuple, is_number, uuid, ctext
from odin.utils.cache_utils import cache_memory
from odin.backend.role import (add_roles, Auxiliary, LearningRate, OptimizerHyperParameter,
                               GradientsNorm, GraidentsClippingNorm, GraidentsClippingValue,
                               has_roles, get_roles,
                               Role, Parameter, Weight, Bias, TrainableParameter,
                               Variable as _Variable, OptimizerVariable)
from odin.backend.helpers import (is_tensor, get_value, ComputationGraph, is_variable,
                                  get_all_variables, set_value)

floatX = CONFIG.floatX

__all__ = [
    "Optimizer",
    "SGD",
    "RMSProp",
    "Adadelta",
    "Adam",
    "Adamax",
    "Nadam",
    "Adagrad"
]
# ===========================================================================
# Helper methods
# ===========================================================================
def _as_variable(x, name, roles=None):
  # nothing to do
  if x is None:
    return None
  # create variable
  if not is_tensor(x):
    x = tf.Variable(x, dtype=floatX, name=name)
    get_session().run(x.initializer)
  return add_roles(x, roles)

# ===========================================================================
# Optimizer
# ===========================================================================
@add_metaclass(ABCMeta)
class Optimizer(object):

  """
  Parameters
  ----------
  lr: float, variable
      learning rate
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Note
  ----
  decayed_learning_rate = learning_rate *
                      decay_rate ^ (global_step / decay_steps)
  """

  def __init__(self, lr, decay_steps=None, decay_rate=0.96, staircase=True,
               clipnorm=None, clipvalue=None, clip_alg='total_norm',
               name=None):
    if name is None:
      name = self.__class__.__name__ + '_' + str(uuid(length=4))
    elif not isinstance(name, string_types):
      name = str(name)
    self._name = str(name)
    self.staircase = bool(staircase)
    with tf.variable_scope(self._name):
      self._lr = _as_variable(lr, name='learning_rate', roles=LearningRate)
      self._lr_decay = None
      self._step = tf.Variable(0., dtype=floatX,
          name="%s_step" % self.__class__.__name__)
      self.decay_steps = decay_steps
      self.decay_rate = decay_rate

      if clipnorm is not None:
        if (clipnorm if is_number(clipnorm) else get_value(clipnorm)) <= 0:
          raise ValueError('`clipnorm` value must greater than 0.')
      self.clipnorm = _as_variable(clipnorm, name="clip_norm",
          roles=GraidentsClippingNorm)

      if clipvalue is not None:
        if (clipvalue if is_number(clipvalue) else get_value(clipvalue)) <= 0:
          raise ValueError('`clipvalue` value must greater than 0.')
      self.clipvalue = _as_variable(clipvalue, name="clip_value",
          roles=GraidentsClippingValue)
    # ====== internal states values ====== #
    clip_alg = str(clip_alg).strip().lower()
    if clip_alg not in ('total_norm', 'norm', 'avg_norm'):
      raise ValueError("clip_arg must be one of the following: "
          "'norm', 'total_norm', 'avg_norm'")
    self._norm = 0.
    self.clip_alg = clip_alg
    self._algorithm = None
    self._is_initialized = False

  @property
  def name(self):
    return self._name

  @property
  def step(self):
    return self._step

  @property
  def lr(self):
    """ Return exponential_decay LR if the decay_steps is given
    otherwise, return the lr variable.
    """
    if self.decay_steps is not None:
      if self._lr_decay is None:
        self._lr_decay = tf.train.exponential_decay(self._lr,
            self._step, self.decay_steps, self.decay_rate,
            staircase=self.staircase)
      return self._lr_decay
    else:
      return self._lr

  @property
  def algorithm(self):
    return self._algorithm

  @property
  def norm(self):
    """Return L2-norm value of all gradients """
    return self._norm

  def minimize(self, loss, var_list=None,
               roles=[TrainableParameter], exclude_roles=[],
               reuse_scope=None, verbose=False):
    """
    Parameters
    ----------
    loss : {Tensor, list of Tensor}
      the loss for minimize
    var_list : {None, list of tensorflow.Variable}
      if None, automatically selecting the Variables, by getting
      all Variables associated with `loss` and using given
      `roles` and `exclude_roles`
    roles : {None, list of odin.backend.role}
      in case of `var_list=None`, all Variables with given role
      in this list will be selected for calculating gradients.
    exclude_roles : {None, list of odin.backend.role}
      all Variables have role in this list will be ignored.
    reuse_scope : {None, string}
      if provided, searching for all `OptimizerVariable` within
      given scope, and set the variables of this optimizer to
      the desire values
    verbose : bool (default: False)
      if True, print out all found variables and their roles
    """
    if exclude_roles is None:
      exclude_roles = []
    # ====== get all relevant variables ====== #
    if var_list is not None:
      all_variables = as_tuple(var_list, t=is_variable)
    else:
      all_variables = ComputationGraph(loss).variables
    # ====== filtering by Roles ====== #
    trainable = [v for v in all_variables
                 if has_roles(v, roles=roles)]
    if isinstance(exclude_roles, Role) or len(exclude_roles) > 0:
      trainable = [v for v in trainable
                   if not has_roles(v, roles=exclude_roles)]
    # ====== filtering by dtype ====== #
    # remove all boolean and string dtype
    trainable = [v
                 for v in trainable
                 if 'int' in str(v.dtype) or 'float' in str(v.dtype)]
    # ====== verbose ====== #
    if bool(verbose):
      # print loss first
      print("Loss:", ctext(str(loss), 'yellow'))
      # organize variable by role
      role_vars = defaultdict(list)
      for var in trainable:
        # remove all tensorflow collection name string
        roles = [r
                 for r in get_roles(var, return_string=False)
                 if not isinstance(r, string_types)]
        # only select highest role in the hierarchy
        high_roles = []
        for r1 in roles:
          found_ancester = False
          for r2 in roles:
            if r1 != r2 and issubclass(r2, r1):
              found_ancester = True
          if not found_ancester:
            high_roles.append(r1)
        # append to mapping role -> var_list
        for r in roles:
          role_vars[r.__name__].append(var)
      # print debug info (this is ugly but look nice and fun)
      if len(role_vars) > 0:
        max_name_length = str(max(
            max(len(i.name) for i in v)
            for v in role_vars.values()))
        max_shape_length = str(max(
            max(len(str(i.shape.as_list())) for i in v)
            for v in role_vars.values()))

        for role, var_list in sorted(role_vars.items(),
                                     key=lambda x: str(x[0])):
          print('Role:', ctext(role, 'yellow'))
          for var in sorted(var_list, key=lambda x: x.name):
            print(' ',
                  ('%-' + max_name_length + 's') % var.name.replace('/', ' '),
                  ctext(('%-' + max_shape_length + 's') % var.shape.as_list(), 'magenta'),
                  ctext(var.dtype.base_dtype.name, 'cyan'))
    # ====== get the updates ====== #
    updates = self.get_updates(loss_or_grads=loss, params=trainable)
    # with tf.variable_scope(self.name):
    #   updates = self.algorithm.minimize(loss=loss, global_step=self._step,
    #                                     var_list=trainable)
    # ====== re-use variable from previous scope ====== #
    if isinstance(reuse_scope, string_types):
      old_variables = [v for v in get_all_variables(scope=reuse_scope)
                       if has_roles(v, OptimizerVariable)]
      old_variables = sorted(old_variables,
                             key=lambda x: x.name.replace(reuse_scope, ''))
      old_varname = [v.name.replace(reuse_scope, '') for v in old_variables]
      # the hyper-parameters may not be included in `old_variables`
      # so they are needed to be excluded in new_variables
      new_variables = [v for v in get_all_variables(scope=self.name)
                       if has_roles(v, OptimizerVariable)]
      new_variables = sorted(new_variables,
                             key=lambda x: x.name.replace(self.name, ''))
      new_variables = [v for v in new_variables
                       if v.name.replace(self.name, '') in old_varname]
      assert len(old_variables) == len(new_variables), \
      "Number of variables in scope:'%s' is %d mismatch scope:'%s' with %d variables"\
      % (reuse_scope, len(old_variables), self.name, len(new_variables))
      for v_old, v_new in zip(old_variables, new_variables):
        assert v_new.shape == v_old.shape
        set_value(v_new, get_value(v_old), return_ops=False)
    return updates

  @cache_memory
  def get_updates(self, loss_or_grads, params):
    grads_vars = self.get_gradients(loss_or_grads, params)
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
      scope_name = scope.name
      updates = self.algorithm.apply_gradients(grads_vars,
                                               global_step=self._step)
    for v in get_all_variables(scope=scope_name):
      add_roles(v, roles=OptimizerVariable)
    return updates

  def __call__(self, loss_or_grads, params):
    return self.get_updates(loss_or_grads, params)

  def grad(self, loss_or_grads, params):
    return self.get_gradients(loss_or_grads=loss_or_grads,
                              params=params)

  @cache_memory
  def get_gradients(self, loss_or_grads, params):
    """
    Note
    ----
    The returned gradients may contain None value
    """
    # check valid algorithm
    if self.algorithm is None or \
    not hasattr(self.algorithm, 'compute_gradients') or \
    not hasattr(self.algorithm, 'apply_gradients'):
      raise RuntimeError("Optimizer is None, or doesn't has attributes: "
                         "compute_gradients and apply_gradients.")
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
      scope_name = scope.name
      # get the gradient
      grads_var = self.algorithm.compute_gradients(loss_or_grads,
                                                   var_list=params)
      grads_var = {g: v for g, v in grads_var if g is not None}
      grads = list(grads_var.keys())
      params = list(grads_var.values())
      # ====== clipnorm ====== #
      if self.clipnorm is not None:
        if self.clip_alg == 'norm':
          grads = [tf.clip_by_norm(g, self.clipnorm)
                   for g in grads]
        elif self.clip_alg == 'total_norm':
          grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
        elif self.clip_alg == 'avg_norm':
          grads = [tf.clip_by_average_norm(g, self.clipnorm)
                   for g in grads]
        else:
          raise ValueError("Unknown norm clipping algorithm: '%s'" % self.clip_alg)
      # ====== clipvalue ====== #
      if self.clipvalue is not None:
        grads = [tf.clip_by_value(g, -self.clipvalue, self.clipvalue)
                 for g in grads]
      # ====== get final norm value ====== #
      self._norm = add_roles(tf.global_norm(grads, name="GradientNorm"),
                             GradientsNorm)
    # ====== setting Optimizer roles ====== #
    for v in get_all_variables(scope=scope_name):
      add_roles(v, roles=OptimizerVariable)
    return [(g, p) for g, p in zip(grads, params)]

class SGD(Optimizer):

  """

  Parameters
  ----------
  momentum: float >= 0. None
      Parameter updates momentum. If momentum is None or 0,
      no momentum is not applied
  nesterov: boolean.
      Whether to apply Nesterov momentum.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  """

  def __init__(self, lr=0.01, momentum=0.9, nesterov=False,
               decay_steps=None, decay_rate=0.96, staircase=True,
               clipnorm=None, clipvalue=None,
               clip_alg='total_norm', name=None):
    super(SGD, self).__init__(lr=lr,
        decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase,
        clipnorm=clipnorm, clipvalue=clipvalue, clip_alg=clip_alg,
        name=name)
    with tf.variable_scope(self.name):
      # ====== momentum ====== #
      self.momentum = _as_variable(momentum, name='momentum',
                                   roles=OptimizerHyperParameter)
      self.nesterov = nesterov
      # ====== decay ====== #
      if self.momentum is not None:
        self._algorithm = tf.train.MomentumOptimizer(self.lr,
            self.momentum, use_nesterov=nesterov)
      else:
        self._algorithm = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr)


class RMSProp(Optimizer):
  """RMSProp updates

  Scale learning rates by dividing with the moving average of the root mean
  squared (RMS) gradients. See [1]_ for further description.

  Parameters
  ----------
  lr : float or symbolic scalar
      The learning rate controlling the size of update steps
  rho : float or symbolic scalar
      Gradient moving average decay factor
  epsilon : float or symbolic scalar
      Small value added for numerical stability, `epsilon` might
      have huge impact on final performance.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average
  fast.

  Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
  learning rate :math:`\\eta_t` is calculated as:

  .. math::
     r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
     \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

  References
  ----------
  .. [1] Tieleman, T. and Hinton, G. (2012):
         Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
         Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
  """

  def __init__(self, lr=0.001, rho=0.9, momentum=0.0, epsilon=1e-10,
               decay_steps=None, decay_rate=0.96, staircase=True,
               clipnorm=None, clipvalue=None, clip_alg='total_norm',
               name=None):
    super(RMSProp, self).__init__(lr=lr,
        decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase,
        clipnorm=clipnorm, clipvalue=clipvalue, clip_alg=clip_alg,
        name=name)
    with tf.variable_scope(self.name):
      self.rho = _as_variable(rho, name='rho',
                              roles=OptimizerHyperParameter)
      self.momentum = _as_variable(momentum, name='momentum',
                              roles=OptimizerHyperParameter)
      self.epsilon = epsilon
      self._algorithm = tf.train.RMSPropOptimizer(self.lr,
          decay=self.rho, momentum=self.momentum, epsilon=self.epsilon,
          centered=False)


class Adadelta(Optimizer):
  """ Adadelta updates

  Scale learning rates by the ratio of accumulated gradients to accumulated
  updates, see [1]_ and notes for further description.

  Parameters
  ----------
  lr : float or symbolic scalar
      The learning rate controlling the size of update steps
  rho : float or symbolic scalar
      Squared gradient moving average decay factor
  epsilon : float or symbolic scalar
      Small value added for numerical stability, `epsilon` might
      have huge impact on final performance.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  rho should be between 0 and 1. A value of rho close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average
  fast.

  rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
  work for multiple datasets (MNIST, speech).

  In the paper, no learning rate is considered (so learning_rate=1.0).
  Probably best to keep it at this value.
  epsilon is important for the very first update (so the numerator does
  not become 0).

  Using the step size eta and a decay factor rho the learning rate is
  calculated as:

  .. math::
     r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
     \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                           {\sqrt{r_t + \epsilon}}\\\\
     s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2

  References
  ----------
  .. [1] Zeiler, M. D. (2012):
         ADADELTA: An Adaptive Learning Rate Method.
         arXiv Preprint arXiv:1212.5701.
  """

  def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8,
               decay_steps=None, decay_rate=0.96, staircase=True,
               clipnorm=None, clipvalue=None, clip_alg='total_norm',
               name=None):
    super(Adadelta, self).__init__(lr=lr,
        decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase,
        clipnorm=clipnorm, clipvalue=clipvalue, clip_alg=clip_alg,
        name=name)
    with tf.variable_scope(self.name):
      self.rho = _as_variable(rho, name='rho',
                              roles=OptimizerHyperParameter)
      self.epsilon = epsilon
      self._algorithm = tf.train.AdadeltaOptimizer(
          learning_rate=self.lr, rho=self.rho,
          epsilon=self.epsilon)

class Adam(Optimizer):
  """Adam updates

  Adam updates implemented as in [1]_.

  Parameters
  ----------
  lr : float
      Learning rate
  beta1 : float
      Exponential decay rate for the first moment estimates.
  beta2 : float
      Exponential decay rate for the second moment estimates.
  epsilon : float or symbolic scalar
      Small value added for numerical stability, `epsilon` might
      have huge impact on final performance.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  The paper [1]_ includes an additional hyperparameter lambda. This is only
  needed to prove convergence of the algorithm and has no practical use
  (personal communication with the authors), it is therefore omitted here.

  References
  ----------
  .. [1] Kingma, Diederik, and Jimmy Ba (2014):
         Adam: A Method for Stochastic Optimization.
         arXiv preprint arXiv:1412.6980.
  """

  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               decay_steps=None, decay_rate=0.96, staircase=True,
               clipnorm=None, clipvalue=None, clip_alg='total_norm',
               name=None):
    super(Adam, self).__init__(lr=lr,
        decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase,
        clipnorm=clipnorm, clipvalue=clipvalue, clip_alg=clip_alg,
        name=name)
    with tf.variable_scope(self.name):
      self.beta1 = _as_variable(beta1, name='beta1',
                                roles=OptimizerHyperParameter)
      self.beta2 = _as_variable(beta2, name='beta2',
                                roles=OptimizerHyperParameter)
      self.epsilon = epsilon
      self._algorithm = tf.train.AdamOptimizer(learning_rate=self.lr,
          beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)

class Adamax(Optimizer):
  """Adamax updates

  Adamax updates implemented as in [1]_. This is a variant of of the Adam
  algorithm based on the infinity norm.

  Parameters
  ----------
  lr : float
      Learning rate
  beta1 : float
      Exponential decay rate for the first moment estimates.
  beta2 : float
      Exponential decay rate for the weighted infinity norm estimates.
  epsilon : float or symbolic scalar
      Small value added for numerical stability, `epsilon` might
      have huge impact on final performance.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  References
  ----------
  .. [1] Kingma, Diederik, and Jimmy Ba (2014):
         Adam: A Method for Stochastic Optimization.
         arXiv preprint arXiv:1412.6980.
  """

  def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
               clipnorm=None, clipvalue=None, clip_alg='total_norm',
               name=None):
    super(Adamax, self).__init__(lr, clipnorm, clipvalue, clip_alg=clip_alg,
                                 name=name)
    with tf.variable_scope(self.name):
      self.iterations = _as_variable(0, name='iterations', roles=Auxiliary)
      self.beta_1 = _as_variable(beta_1, name='beta_1',
                                 roles=OptimizerHyperParameter)
      self.beta_2 = _as_variable(beta_2, name='beta_2',
                                 roles=OptimizerHyperParameter)
      self.epsilon = epsilon
      raise NotImplementedError


class Nadam(Optimizer):
  """Nadam updates

  Nesterov Adam optimizer: Much like Adam is essentially RMSprop with momentum,
  Nadam is Adam RMSprop with Nesterov momentum.

  Parameters
  ----------
  lr : float
      Learning rate
  beta1 : float
      Exponential decay rate for the first moment estimates.
  beta2 : float
      Exponential decay rate for the second moment estimates.
  epsilon : float or symbolic scalar
      Small value added for numerical stability, `epsilon` might
      have huge impact on final performance.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  The paper [1]_ includes an additional hyperparameter lambda. This is only
  needed to prove convergence of the algorithm and has no practical use
  (personal communication with the authors), it is therefore omitted here.

  References
  ----------
  .. [1] [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
  .. [2] [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

  """

  def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
               schedule_decay=0.004, clipnorm=None, clipvalue=None,
               clip_alg='total_norm', name=None):
    super(Nadam, self).__init__(lr, clipnorm, clipvalue, clip_alg=clip_alg,
                                name=name)
    with tf.variable_scope(self.name):
      self.iterations = _as_variable(0, name='iterations', roles=Auxiliary)
      self.m_schedule = _as_variable(1., name='m_schedule', roles=Auxiliary)
      self.schedule_decay = schedule_decay
      self.beta_1 = _as_variable(beta_1, name='beta_1',
                                 roles=OptimizerHyperParameter)
      self.beta_2 = _as_variable(beta_2, name='beta_2',
                                 roles=OptimizerHyperParameter)
      self.epsilon = epsilon
      raise NotImplementedError


class Adagrad(Optimizer):
  """Adagrad updates

  Scale learning rates by dividing with the square root of accumulated
  squared gradients. See [1]_ for further description.

  Parameters
  ----------
  lr : float or symbolic scalar
      The learning rate controlling the size of update steps
  initial_accumulator_value: A floating point value.
      Starting value for the gradients accumulators, must be positive.
  dual_avg: bool
      if True, use Adagrad Dual Averaging algorithm for sparse linear models.
  decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive. (e.g. decay every 100000 steps with a base of 0.96)
  decay_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
  staircase: boolean.
      If `True` decay the learning rate at discrete intervals
  clipnorm: float >= 0. Gradients will be clipped
      when their L2 norm exceeds this value.
  clipvalue: float >= 0. Gradients will be clipped
      when their absolute value exceeds this value.
  clip_alg: str
      clipping algorithm for the gradients, one of the followings are
      accepted: "norm", "total_norm", "avg_norm"

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  Using step size eta Adagrad calculates the learning rate for feature i at
  time step t as:

  .. math:: \\eta_{t,i} = \\frac{\\eta}
     {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}

  as such the learning rate is monotonically decreasing.

  Epsilon is not included in the typical formula, see [2]_.

  References
  ----------
  .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
         Adaptive subgradient methods for online learning and stochastic
         optimization. JMLR, 12:2121-2159.

  .. [2] Chris Dyer:
         Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
  """

  def __init__(self, lr=0.01, initial_accumulator_value=0.1,
               dual_avg=False,
               l1_regularization=0.0, l2_regularization=0.0,
               decay_steps=None, decay_rate=0.96, staircase=True,
               clipnorm=None, clipvalue=None, clip_alg='total_norm',
               name=None):
    super(Adagrad, self).__init__(lr=lr,
        decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase,
        clipnorm=clipnorm, clipvalue=clipvalue, clip_alg=clip_alg,
        name=name)
    with tf.variable_scope(self.name):
      self.initial_accumulator_value = float(initial_accumulator_value)
      self.l1_regularization = _as_variable(l1_regularization,
          name='l1_regularization', roles=OptimizerHyperParameter)
      self.l2_regularization = _as_variable(l2_regularization,
          name='l2_regularization', roles=OptimizerHyperParameter)
      self.dual_avg = bool(dual_avg)
      # create algorithm
      if self.dual_avg:
        self._algorithm = tf.train.AdagradDAOptimizer(learning_rate=self.lr,
            global_step=self._step,
            initial_gradient_squared_accumulator_value=self.initial_accumulator_value,
            l1_regularization_strength=self.l1_regularization,
            l2_regularization_strength=self.l2_regularization)
      else:
        self._algorithm = tf.train.AdagradOptimizer(learning_rate=self.lr,
            initial_accumulator_value=self.initial_accumulator_value)
