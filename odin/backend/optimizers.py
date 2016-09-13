# ===========================================================================
# This module is created based on the code from libraries: Lasagne
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import numbers
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from six import add_metaclass

import numpy as np

from odin.config import autoconfig
from odin.utils import as_tuple
from odin.roles import add_roles, AUXILIARY, LEARNING_RATE

FLOATX = autoconfig.floatX

# store python primitive operators
_sum = sum
_pow = pow
_abs = abs

if autoconfig['backend'] == 'theano':
    from .theano import (is_variable, is_trainable_variable, is_placeholder,
                         gradients, get_shape, switch, get_value,
                         variable, constant, cast, square,
                         sqrt, maximum, abs, clip)
elif autoconfig['backend'] == 'tensorflow':
    from .tensorflow import (is_variable, is_placeholder, gradients)

__all__ = [
    "sgd",
    "apply_momentum",
    "momentum",
    "apply_nesterov_momentum",
    "nesterov_momentum",
    "adagrad",
    "rmsprop",
    "adadelta",
    "adam",
    "adamax",
    "norm_constraint",
    "total_norm_constraint"
]


# ===========================================================================
# Helper methods
# ===========================================================================
def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for

    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.

    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    if any(not (is_variable(p) or is_placeholder(p)) for p in params):
        raise ValueError("params must contain shared variables only.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return gradients(loss_or_grads, params)


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
    add_roles(norm, AUXILIARY)
    tensor_vars = [switch(norm >= max_norm, g * max_norm / norm, g)
                   for g in tensor_vars]
    # ====== return norm if necessary ====== #
    if return_norm:
        return tensor_vars, norm
    else:
        return tensor_vars


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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

    """

    def __init__(self, lr, clipnorm=None, clipvalue=None):
        if not is_variable(lr):
            self.lr = variable(lr, name='learning_rate')
        add_roles(self.lr, LEARNING_RATE)

        if clipnorm is not None and \
        (clipnorm if isinstance(clipnorm, numbers.Number) else get_value(clipnorm)) <= 0:
            raise ValueError('clipnorm value must greater than 0.')
        self.clipnorm = clipnorm
        if clipvalue is not None and \
        (clipvalue if isinstance(clipnorm, numbers.Number) else get_value(clipvalue)) <= 0:
            raise ValueError('clipvalue value must greater than 0.')
        self.clipvalue = clipvalue
        # ====== internal states values ====== #
        self._norm = 0.
        self._last_updates = {}

    @property
    def norm(self):
        return self._norm

    @property
    def last_updates(self):
        return self._last_updates

    @abstractmethod
    def get_updates(self, loss_or_grads, params):
        pass

    def __call__(self, loss_or_grads, params):
        updates = self.get_updates(loss_or_grads, params)
        if not isinstance(updates, (dict, list, tuple)):
            raise ValueError('returned "updates" must be dict, list, tuple which '
                            'contain pair of (params<=>new_params).')
        self._last_updates = updates
        return updates

    def get_gradients(self, loss_or_grads, params):
        grads = get_or_compute_grads(loss_or_grads, params)
        if self.clipnorm is not None and self.clipnorm > 0:
            grads, self._norm = total_norm_constraint(grads, self.clipnorm,
                                                      return_norm=True)
        if self.clipvalue is not None and self.clipvalue > 0:
            grads = [clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads


class SGD(Optimizer):

    def get_updates(self, loss_or_grads, params):
        pass


def sgd(loss_or_grads, params, learning_rate=0.1):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates


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


def momentum(loss_or_grads, params, learning_rate=0.1, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    See Also
    --------
    apply_momentum : Generic function applying momentum to updates
    nesterov_momentum : Nesterov's variant of SGD with momentum
    """
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_momentum(updates, momentum=momentum)


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


def nesterov_momentum(loss_or_grads, params, learning_rate=0.1, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

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
    apply_nesterov_momentum : Function applying momentum to updates
    """
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)


def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    """Adagrad updates

    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability

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

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        shape = get_shape(param)

        accu = variable(np.zeros(shape))
        # accu = addbroadcast(accu, *broadcastable(param))

        accu_new = accu + pow(grad, 2)
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  sqrt(accu_new + epsilon))

    return updates


def rmsprop(loss_or_grads, params, lr=0.001, rho=0.9, epsilon=1e-6):
    """RMSProp updates

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability

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
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    one = constant(1)

    for param, grad in zip(params, grads):
        shape = get_shape(param)

        accu = variable(np.zeros(shape))
        # accu = addbroadcast(accu, *broadcastable(param))

        accu_new = rho * accu + (one - rho) * pow(grad, 2)
        updates[accu] = accu_new
        updates[param] = param - (lr * grad /
                                  sqrt(accu_new + epsilon))

    return updates


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """ Adadelta updates

    Scale learning rates by the ratio of accumulated gradients to accumulated
    updates, see [1]_ and notes for further description.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Squared gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability

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
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    one = constant(1)

    for param, grad in zip(params, grads):
        shape = get_shape(param)
        # accu: accumulate gradient magnitudes
        accu = variable(np.zeros(shape))
        # accu = addbroadcast(accu, *broadcastable(param))
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = variable(np.zeros(shape))
        # delta_accu = addbroadcast(delta_accu, *broadcastable(param))
        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * pow(grad, 2)
        updates[accu] = accu_new
        # compute parameter update, using the 'old' delta_accu
        update = (grad * sqrt(delta_accu + epsilon) /
                  sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update
        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * pow(update, 2)
        updates[delta_accu] = delta_accu_new

    return updates


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """Adam updates

    Adam updates implemented as in [1]_.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float
        Learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.

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
    # TODO: check again this adam
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = variable(0.)
    updates = OrderedDict()
    one = constant(1)

    t = t_prev + 1
    a_t = learning_rate * \
        sqrt(one - _pow(cast(beta2, FLOATX), t)) / (one - _pow(cast(beta1, FLOATX), t))

    for param, g_t in zip(params, all_grads):
        shape = get_shape(param)

        m_prev = variable(np.zeros(shape))
        # m_prev = addbroadcast(m_prev, *broadcastable(param))

        v_prev = variable(np.zeros(shape))
        # v_prev = addbroadcast(v_prev, *broadcastable(param))

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * pow(g_t, 2)
        step = a_t * m_t / (sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates


def adamax(loss_or_grads, params, learning_rate=0.002, beta1=0.9,
           beta2=0.999, epsilon=1e-8):
    """Adamax updates

    Adamax updates implemented as in [1]_. This is a variant of of the Adam
    algorithm based on the infinity norm.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float
        Learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the weighted infinity norm estimates.
    epsilon : float
        Constant for numerical stability.

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
    all_grads = get_or_compute_grads(loss_or_grads, params)
    t_prev = variable(0.)
    updates = OrderedDict()
    one = constant(1)

    t = t_prev + 1
    a_t = learning_rate / (one - pow(cast(beta1, FLOATX), t))

    for param, g_t in zip(params, all_grads):
        shape = get_shape(param)

        m_prev = variable(np.zeros(shape))
        # m_prev = addbroadcast(m_prev, *broadcastable(param))

        u_prev = variable(np.zeros(shape))
        # u_prev = addbroadcast(u_prev, *broadcastable(param))

        m_t = beta1 * m_prev + (one - beta1) * g_t
        u_t = maximum(beta2 * u_prev, abs(g_t))
        step = a_t * m_t / (u_t + epsilon)

        updates[m_prev] = m_t
        updates[u_prev] = u_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates
