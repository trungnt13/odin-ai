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

from odin.config import CONFIG
from odin.utils import as_tuple
from odin.basic import (add_role, AUXILIARY, LEARNING_RATE,
                        OPTIMIZER_HYPER_PARAMETER, VariableRole)
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
        roles = as_tuple(roles, t=VariableRole)
        for r in roles:
            add_role(x, r)
    return x


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
    add_role(norm, AUXILIARY)
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
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

    """

    def __init__(self, lr, clipnorm=None, clipvalue=None):
        self.lr = _as_variable(lr, name='learning_rate', roles=LEARNING_RATE)

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
        self.updates = {}

    @property
    def norm(self):
        """Return L2-norm value of all gradients """
        return self._norm

    @abstractmethod
    def get_updates(self, loss_or_grads, params):
        pass

    def __call__(self, loss_or_grads, params):
        updates = self.get_updates(loss_or_grads, params)
        if not isinstance(updates, (dict, list, tuple)):
            raise ValueError('returned "updates" must be dict, list, tuple which '
                            'contain pair of (params<=>new_params).')
        self.updates = updates
        return updates

    def get_gradients(self, loss_or_grads, params):
        """
        Note
        ----
        The returned gradients may contain None value
        """
        grads = get_or_compute_grads(loss_or_grads, params)
        # ====== clipnorm ====== #
        if self.clipnorm is not None and self.clipnorm > 0:
            grads, self._norm = total_norm_constraint(grads, self.clipnorm,
                                                      return_norm=True)
        else:
            self._norm = sqrt(_sum([sum(square(g)) for g in grads
                                    if g is not None]))
            add_role(self._norm, AUXILIARY)
        # ====== clipvalue ====== #
        if self.clipvalue is not None and self.clipvalue > 0:
            grads = [clip(g, -self.clipvalue, self.clipvalue)
                     if g is not None else g
                     for g in grads]
        return grads

    def get_lr_callback(self, decay=2.):
        """ Return: a lambda function, everytime you call this function
        the learning rate is decayed by given factor.
        :math:`lr_{new} = lr_{old} / decay`
        """
        def lr_decay():
            lr = get_value(self.lr)
            lr = lr / decay
            set_value(self.lr, lr)

        return lr_decay


class SGD(Optimizer):

    """
    Parameters
    ----------
    momentum: float >= 0. None
        Parameter updates momentum. If momentum is None or 0,
        no momentum is not applied
    decay: float >= 0, or None
        Learning rate decay over each update. If decay is None or 0,
        no decay is applied
    nesterov: boolean.
        Whether to apply Nesterov momentum.
    clipnorm: float >= 0. Gradients will be clipped
        when their L2 norm exceeds this value.
    clipvalue: float >= 0. Gradients will be clipped
        when their absolute value exceeds this value.

    """

    def __init__(self, lr=0.01, momentum=0.9, decay=None, nesterov=False,
                 clipnorm=None, clipvalue=None):
        super(SGD, self).__init__(lr, clipnorm, clipvalue)
        self.iterations = _as_variable(0., name='iterations', roles=AUXILIARY)
        self.nesterov = nesterov
        # ====== momentum ====== #
        if momentum == 0:
            momentum = None
        self.momentum = _as_variable(momentum, name='momentum',
                                     roles=OPTIMIZER_HYPER_PARAMETER)
        # ====== decay ====== #
        if decay == 0:
            decay = None
        self.decay = _as_variable(decay, name='decay',
                                  roles=OPTIMIZER_HYPER_PARAMETER)

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)
        lr = self.lr
        updates = []
        if self.decay is not None:
            lr = lr * (1. / (1. + self.decay * self.iterations))
            updates.append((self.iterations, self.iterations + 1))

        # momentum
        if self.momentum is not None:
            shapes = [get_shape(p) for p in params]
            moments = [variable(np.zeros(shape)) for shape in shapes]
        else: # just create dummy moments
            moments = [None] * len(params)
        # ====== main updates ====== #
        for p, g, m in zip(params, grads, moments):
            if g is None: continue
            update_gradient = lr * g
            # ====== applying momentum ====== #
            if self.momentum is not None:
                v = self.momentum * m - update_gradient  # velocity
                updates.append((m, v))
                if self.nesterov:
                    new_p = p + self.momentum * v - update_gradient
                else:
                    new_p = p + v
            # ====== NO momentum ====== #
            else:
                new_p = p - update_gradient
            # final updates
            updates.append((p, new_p))
        return updates


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

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8,
                 clipnorm=None, clipvalue=None):
        super(RMSProp, self).__init__(lr, clipnorm, clipvalue)
        self.rho = _as_variable(rho, name='rho',
                                roles=OPTIMIZER_HYPER_PARAMETER)
        self.epsilon = constant(epsilon, dtype=FLOATX)

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)

        shapes = [get_shape(p) for p in params]
        accumulators = [variable(np.zeros(shape)) for shape in shapes]
        updates = []

        for p, g, a in zip(params, grads, accumulators):
            if g is None: continue
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * square(g)
            updates.append((a, new_a))
            # update parameters
            new_p = p - self.lr * g / (sqrt(new_a) + self.epsilon)
            # add to updates
            updates.append((p, new_p))
        return updates


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
                 clipnorm=None, clipvalue=None):
        super(Adadelta, self).__init__(lr, clipnorm, clipvalue)
        self.rho = _as_variable(rho, name='rho',
                                roles=OPTIMIZER_HYPER_PARAMETER)
        self.epsilon = epsilon

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)

        shapes = [get_shape(p) for p in params]
        accumulators = [variable(np.zeros(shape)) for shape in shapes]
        delta_accumulators = [variable(np.zeros(shape)) for shape in shapes]
        updates = []

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            if g is None: continue
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * square(g)
            updates.append((a, new_a))
            # use the new accumulator and the *old* delta_accumulator
            update = g * sqrt(d_a + self.epsilon) / sqrt(new_a + self.epsilon)
            # update new parameters
            new_p = p - self.lr * update
            updates.append((p, new_p))
            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * square(update)
            updates.append((d_a, new_d_a))
        return updates


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

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 clipnorm=None, clipvalue=None):
        super(Adam, self).__init__(lr, clipnorm, clipvalue)
        self.iterations = _as_variable(0, name='iterations', roles=AUXILIARY)
        self.beta_1 = _as_variable(beta_1, name='beta_1',
                                   roles=OPTIMIZER_HYPER_PARAMETER)
        self.beta_2 = _as_variable(beta_2, name='beta_2',
                                   roles=OPTIMIZER_HYPER_PARAMETER)
        self.epsilon = epsilon

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)
        t = self.iterations + 1
        updates = [(self.iterations, t)]

        lr_t = self.lr * sqrt(1. - pow(self.beta_2, t)) / (1. - pow(self.beta_1, t))

        shapes = [get_shape(p) for p in params]
        ms = [variable(np.zeros(shape)) for shape in shapes]
        vs = [variable(np.zeros(shape)) for shape in shapes]

        for p, g, m, v in zip(params, grads, ms, vs):
            if g is None: continue
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * square(g)
            p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)
            # updates new statistics
            updates.append((m, m_t))
            updates.append((v, v_t))
            # updates new params
            new_p = p_t
            updates.append((p, new_p))
        return updates


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
        self.iterations = _as_variable(0, name='iterations', roles=AUXILIARY)
        self.beta_1 = _as_variable(beta_1, name='beta_1',
                                   roles=OPTIMIZER_HYPER_PARAMETER)
        self.beta_2 = _as_variable(beta_2, name='beta_2',
                                   roles=OPTIMIZER_HYPER_PARAMETER)
        self.epsilon = epsilon

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)
        t = self.iterations + 1
        updates = [(self.iterations, t)]

        lr_t = self.lr / (1. - pow(self.beta_1, t))

        shapes = [get_shape(p) for p in params]
        # zero init of 1st moment
        ms = [variable(np.zeros(shape)) for shape in shapes]
        # zero init of exponentially weighted infinity norm
        us = [variable(np.zeros(shape)) for shape in shapes]
        for p, g, m, u in zip(params, grads, ms, us):
            if g is None: continue
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = maximum(self.beta_2 * u, abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)
            updates.append((m, m_t))
            updates.append((u, u_t))
            # updates new parametesr
            new_p = p_t
            updates.append((p, new_p))
        return updates


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
        self.iterations = _as_variable(0, name='iterations', roles=AUXILIARY)
        self.m_schedule = _as_variable(1., name='m_schedule', roles=AUXILIARY)
        self.schedule_decay = schedule_decay

        self.beta_1 = _as_variable(beta_1, name='beta_1',
                                   roles=OPTIMIZER_HYPER_PARAMETER)
        self.beta_2 = _as_variable(beta_2, name='beta_2',
                                   roles=OPTIMIZER_HYPER_PARAMETER)
        self.epsilon = epsilon

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)
        t = self.iterations + 1
        updates = [(self.iterations, t)]

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (pow(0.96, t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (pow(0.96, (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        updates.append((self.m_schedule, m_schedule_new))

        shapes = [get_shape(p) for p in params]
        ms = [variable(np.zeros(shape)) for shape in shapes]
        vs = [variable(np.zeros(shape)) for shape in shapes]

        for p, g, m, v in zip(params, grads, ms, vs):
            if g is None: continue
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * square(g)
            v_t_prime = v_t / (1. - pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            updates.append((m, m_t))
            updates.append((v, v_t))
            # update the parameters
            p_t = p - self.lr * m_t_bar / (sqrt(v_t_prime) + self.epsilon)
            new_p = p_t
            updates.append((p, new_p))
        return updates


class Adagrad(Optimizer):
    """Adagrad updates

    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.

    Parameters
    ----------
    lr : float or symbolic scalar
        The learning rate controlling the size of update steps
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

    def __init__(self, lr=0.01, epsilon=1e-8,
                 clipnorm=None, clipvalue=None):
        super(Adagrad, self).__init__(lr, clipnorm, clipvalue)
        self.epsilon = epsilon

    def get_updates(self, loss_or_grads, params):
        grads = self.get_gradients(loss_or_grads, params)
        updates = []

        shapes = [get_shape(p) for p in params]
        accumulators = [variable(np.zeros(shape)) for shape in shapes]

        for p, g, a in zip(params, grads, accumulators):
            if g is None: continue
            # update accumulator
            new_a = a + square(g)
            updates.append((a, new_a))
            # new parameters
            new_p = p - (self.lr * g / sqrt(new_a + self.epsilon))
            updates.append((p, new_p))
        return updates
