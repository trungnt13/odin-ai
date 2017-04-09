# ===========================================================================
# This module is created based on the code from libraries: Lasagne
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import numbers
from abc import ABCMeta, abstractproperty
from collections import OrderedDict
from six import add_metaclass

import numpy as np

from odin.config import CONFIG
from odin.utils import as_tuple, is_number
from odin.utils.cache_utils import cache_memory
from odin.basic import (add_role, Auxiliary, LearningRate,
                        OptimizerHyperParameter, Role, GradientsNorm)
FLOATX = CONFIG.floatX

# store python primitive operators
_sum = sum
_pow = pow
_abs = abs

from .basic_ops import (is_variable, is_placeholder, gradients, get_shape,
                        switch, set_value, get_value, variable, cast, constant,
                        square, sum, sqrt, maximum, abs, clip)

__all__ = [
    "apply_momentum",
    "apply_nesterov_momentum",
    "total_norm_constraint",

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
    if not is_variable(x):
        x = variable(x, name=name)
    if roles is not None:
        roles = [r for r in as_tuple(roles) if issubclass(r, Role)]
        for r in roles:
            add_role(x, r)
    return x


def total_norm_constraint(tensor_vars, max_norm, epsilon=1e-8,
                          return_norm=False):
    """Rescales a list of tensors based on their combined norm

    If the combined norm of the input tensors exceeds the threshold then all
    tensors are rescaled such that the combined norm is equal to the threshold.

    Scaling the norms of the gradients is often used when training recurrent
    neural networks [1]_.

    Parameters
    ----------
    tensor_vars : List of TensorVariables.
        Tensors to be rescaled.
    max_norm : float
        Threshold value for total norm.
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.
    return_norm : bool
        If true the total norm is also returned.

    Returns
    -------
    tensor_vars_scaled : list of TensorVariables
        The scaled tensor variables.
    norm : Theano scalar
        The combined norms of the input variables prior to rescaling,
        only returned if ``return_norms=True``.

    Notes
    -----
    The total norm can be used to monitor training.

    References
    ----------
    .. [1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014): Sequence to sequence
       learning with neural networks. In Advances in Neural Information
       Processing Systems (pp. 3104-3112).
    """
    max_norm = cast(max_norm, FLOATX)
    epsilon = cast(epsilon, FLOATX)
    tensor_vars = as_tuple(tensor_vars)
    # ====== clip norm ====== #
    norm = sqrt(_sum([sum(square(tensor)) for tensor in tensor_vars]))
    add_role(norm, Auxiliary)
    tensor_vars = [switch(norm >= max_norm, g * max_norm / norm, g)
                   if g is not None else None
                   for g in tensor_vars]
    # ====== return norm if necessary ====== #
    if return_norm:
        return tensor_vars, norm
    else:
        return tensor_vars


def apply_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + velocity``

    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    See Also
    --------
    momentum : Shortcut applying momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        shape = get_shape(param)
        velocity = variable(np.zeros(shape))
        # velocity = addbroadcast(velocity, *broadcastable(param))
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates


def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including Nesterov momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + momentum * velocity + updates[param] - param``

    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.

    See Also
    --------
    nesterov_momentum : Shortcut applying Nesterov momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        shape = get_shape(param)
        velocity = variable(np.zeros(shape))
        # velocity = addbroadcast(velocity, *broadcastable(param))
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates


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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.
    Note
    ----
    decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
    """

    def __init__(self, lr, decay_steps=None, decay_rate=0.96,
                 clipnorm=None, clipvalue=None):
        self._lr = _as_variable(lr, name='learning_rate', roles=LearningRate)
        self._lr_decay = None
        self._step = variable(0., name="%s_step" % self.__class__.__name__)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        if clipnorm is not None and \
        (clipnorm if is_number(clipnorm) else get_value(clipnorm)) <= 0:
            raise ValueError('clipnorm value must greater than 0.')
        self.clipnorm = clipnorm

        if clipvalue is not None and \
        (clipvalue if is_number(clipnorm) else get_value(clipvalue)) <= 0:
            raise ValueError('clipvalue value must greater than 0.')
        self.clipvalue = clipvalue

        # ====== internal states values ====== #
        self._norm = 0.
        self._algorithm = None
        self._is_initialized = False

    @property
    def step(self):
        return self._step

    @property
    def lr_value(self):
        return self._lr

    @property
    def lr(self):
        if self.decay_steps is not None:
            if self._lr_decay is None:
                import tensorflow as tf
                self._lr_decay = tf.train.exponential_decay(self._lr,
                    self._step, self.decay_steps, self.decay_rate,
                    staircase=True)
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

    @cache_memory
    def get_updates(self, loss_or_grads, params):
        grads_vars = self.get_gradients(loss_or_grads, params)
        updates = self.algorithm.apply_gradients(grads_vars,
            global_step=self._step)
        # ====== initialize ====== #
        import tensorflow as tf
        init = tf.global_variables_initializer()
        init.run()
        return updates

    def __call__(self, loss_or_grads, params):
        return self.get_updates(loss_or_grads, params)

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
        # get the gradient
        grads_var = self.algorithm.compute_gradients(loss_or_grads,
            var_list=params)
        grads_var = {g: v for g, v in grads_var if g is not None}
        grads = grads_var.keys()
        params = grads_var.values()
        # ====== clipnorm ====== #
        if self.clipnorm is not None and self.clipnorm > 0:
            grads, self._norm = total_norm_constraint(
                grads, self.clipnorm, return_norm=True)
        else:
            self._norm = sqrt(_sum([sum(square(g)) for g in grads]))
            add_role(self._norm, GradientsNorm)
        # ====== clipvalue ====== #
        if self.clipvalue is not None and self.clipvalue > 0:
            grads = [clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return [(g, p) for g, p in zip(grads, params)]

    def get_lr_callback(self, decay=2.):
        """ Return: a lambda function, everytime you call this function
        the learning rate is decayed by given factor.
        :math:`lr_{new} = lr_{old} / decay`
        """
        def lr_decay():
            lr = get_value(self._lr)
            lr = lr / decay
            set_value(self._lr, lr)

        return lr_decay


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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

    """

    def __init__(self, lr=0.01, momentum=0.9, nesterov=False,
                 decay_steps=None, decay_rate=0.96,
                 clipnorm=None, clipvalue=None):
        super(SGD, self).__init__(lr=lr,
            decay_steps=decay_steps, decay_rate=decay_rate,
            clipnorm=clipnorm, clipvalue=clipvalue)
        # ====== momentum ====== #
        if momentum == 0:
            momentum = None
        self.momentum = _as_variable(momentum, name='momentum',
                                     roles=OptimizerHyperParameter)
        self.nesterov = nesterov
        # ====== decay ====== #
        import tensorflow as tf
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

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
                 decay_steps=None, decay_rate=0.96,
                 clipnorm=None, clipvalue=None):
        super(RMSProp, self).__init__(lr=lr,
            decay_steps=decay_steps, decay_rate=decay_rate,
            clipnorm=clipnorm, clipvalue=clipvalue)
        self.rho = _as_variable(rho, name='rho',
                                roles=OptimizerHyperParameter)
        self.momentum = _as_variable(momentum, name='momentum',
                                roles=OptimizerHyperParameter)
        self.epsilon = epsilon
        import tensorflow as tf
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

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
                 decay_steps=None, decay_rate=0.96,
                 clipnorm=None, clipvalue=None):
        super(Adadelta, self).__init__(lr=lr,
            decay_steps=decay_steps, decay_rate=decay_rate,
            clipnorm=clipnorm, clipvalue=clipvalue)
        self.rho = _as_variable(rho, name='rho',
                                roles=OptimizerHyperParameter)
        self.epsilon = epsilon
        import tensorflow as tf
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

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
                 decay_steps=None, decay_rate=0.96,
                 clipnorm=None, clipvalue=None):
        super(Adam, self).__init__(lr=lr,
            decay_steps=decay_steps, decay_rate=decay_rate,
            clipnorm=clipnorm, clipvalue=clipvalue)
        self.beta1 = _as_variable(beta1, name='beta1',
                                  roles=OptimizerHyperParameter)
        self.beta2 = _as_variable(beta2, name='beta2',
                                  roles=OptimizerHyperParameter)
        self.epsilon = epsilon
        import tensorflow as tf
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

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
                 clipnorm=None, clipvalue=None):
        super(Adamax, self).__init__(lr, clipnorm, clipvalue)
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

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
                 schedule_decay=0.004, clipnorm=None, clipvalue=None):
        super(Nadam, self).__init__(lr, clipnorm, clipvalue)
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

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
                 l1_regularization=0.0,
                 l2_regularization=0.0,
                 decay_steps=None, decay_rate=0.96,
                 clipnorm=None, clipvalue=None):
        super(Adagrad, self).__init__(lr=lr,
            decay_steps=decay_steps, decay_rate=decay_rate,
            clipnorm=clipnorm, clipvalue=clipvalue)
        self.initial_accumulator_value = float(initial_accumulator_value)
        self.l1_regularization = _as_variable(l1_regularization,
            name='l1_regularization')
        self.l2_regularization = _as_variable(l2_regularization,
            name='l2_regularization')
        self.dual_avg = bool(dual_avg)

        import tensorflow as tf
        if self.dual_avg:
            self._algorithm = tf.train.AdagradDAOptimizer(learning_rate=self.lr,
                global_step=self._step,
                initial_gradient_squared_accumulator_value=self.initial_accumulator_value,
                l1_regularization_strength=self.l1_regularization,
                l2_regularization_strength=self.l2_regularization)
        else:
            self._algorithm = tf.train.AdagradOptimizer(learning_rate=self.lr,
                initial_accumulator_value=self.initial_accumulator_value)
