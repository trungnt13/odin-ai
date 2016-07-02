from __future__ import division, absolute_import

from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from odin.config import RNG_GENERATOR, autoconfig
from . import tensor as K

FLOATX = autoconfig.floatX
EPSILON = autoconfig.epsilon
PI = np.pi
C = -0.5 * np.log(2 * PI)
_RNG = RandomStreams(seed=RNG_GENERATOR.randint(10e8))


# ===========================================================================
# Special random algorithm for weights initialization
# ===========================================================================
def normal(shape, mean=0., std=1.):
    return np.cast[FLOATX](
        RNG_GENERATOR.normal(mean, std, size=shape))


def uniform(shape, range=0.05):
    if isinstance(range, (int, float, long)):
        range = (-abs(range), abs(range))
    return np.cast[FLOATX](
        RNG_GENERATOR.uniform(low=range[0], high=range[1], size=shape))


def constant(shape, val=0.):
    return np.cast[FLOATX](np.zeros(shape) + val)


def symmetric_uniform(shape, range=0.01, std=None, mean=0.0):
    if std is not None:
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
    else:
        try:
            a, b = range  # range is a tuple
        except TypeError:
            a, b = -range, range  # range is a number
    return np.cast[FLOATX](
        RNG_GENERATOR.uniform(low=a, high=b, size=shape))


def glorot_uniform(shape, gain=1.0, c01b=False):
    orig_shape = shape
    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
    else:
        if len(shape) < 2:
            shape = (1,) + tuple(shape)
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    a = 0.0 - np.sqrt(3) * std
    b = 0.0 + np.sqrt(3) * std
    return np.cast[FLOATX](
        RNG_GENERATOR.uniform(low=a, high=b, size=orig_shape))


def glorot_normal(shape, gain=1.0, c01b=False):
    orig_shape = shape
    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
    else:
        if len(shape) < 2:
            shape = (1,) + tuple(shape)
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    return np.cast[FLOATX](
        RNG_GENERATOR.normal(0.0, std, size=orig_shape))


def he_normal(shape, gain=1.0, c01b=False):
    if gain == 'relu':
        gain = np.sqrt(2)

    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        fan_in = np.prod(shape[:3])
    else:
        if len(shape) <= 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])

    std = gain * np.sqrt(1.0 / fan_in)
    return np.cast[FLOATX](
        RNG_GENERATOR.normal(0.0, std, size=shape))


def he_uniform(shape, gain=1.0, c01b=False):
    if gain == 'relu':
        gain = np.sqrt(2)

    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        fan_in = np.prod(shape[:3])
    else:
        if len(shape) <= 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])

    std = gain * np.sqrt(1.0 / fan_in)
    a = 0.0 - np.sqrt(3) * std
    b = 0.0 + np.sqrt(3) * std
    return np.cast[FLOATX](
        RNG_GENERATOR.uniform(low=a, high=b, size=shape))


def orthogonal(shape, gain=1.0):
    if gain == 'relu':
        gain = np.sqrt(2)

    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = RNG_GENERATOR.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.cast[FLOATX](gain * q)


# ===========================================================================
# RANDOMNESS
# ===========================================================================
class _RandomWrapper(object):

    def __init__(self, rng):
        super(_RandomWrapper, self).__init__()
        self._rng = rng

    def normal(self, shape, mean, std, dtype=FLOATX):
        return self._rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

    def uniform(self, shape, low, high, dtype=FLOATX):
        return self._rng.uniform(size=shape, low=low, high=high, dtype=dtype)

    def binomial(self, shape, p, dtype=FLOATX):
        return self._rng.binomial(size=shape, n=1, p=p, dtype=dtype)


def rng(seed=None):
    if seed is None:
        seed = RNG_GENERATOR.randint(10e8)
    return _RandomWrapper(RandomStreams(seed=seed))


def random_normal(shape, mean=0.0, std=1.0, dtype=FLOATX, seed=None):
    rng = _RNG
    if seed is not None:
        rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=FLOATX, seed=None):
    rng = _RNG
    if seed is not None:
        rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)


def random_binomial(shape, p, dtype=FLOATX, seed=None):
    rng = _RNG
    if seed is not None:
        rng = RandomStreams(seed=seed)
    return rng.binomial(size=shape, n=1, p=p, dtype=dtype)


# ===========================================================================
# Noise
# ===========================================================================
def _process_noise_dim(input_shape, dims):
    """
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Examples
    --------
    (None, 10, 10) with noise_dims=2
    => (None, 10, 1)
    """
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    # ====== get noise shape ====== #
    if dims is None:
        noise_shape = input_shape
    else:
        return tuple([1 if i in dims else j
                      for i, j in enumerate(input_shape)])
    return noise_shape


def apply_dropout(x, level=0.5, noise_dims=None, rescale=True, seed=None):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.


    Parameters
    ----------
    x: A tensor.
    level: float(0.-1.)
        probability dropout values in given tensor
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    rng: `tensor.rng`
        random generator from tensor class

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """
    input_shape = K.shape(x)
    # ====== not a training variable NO dropout ====== #
    if not K.is_training(x):
        return x
    # ====== Dropout ====== #
    retain_prob = 1. - level
    shape = K.shape(x, none=False)
    if noise_dims is None:
        x = x * K.random_binomial(shape=shape, p=retain_prob, dtype=x.dtype, seed=seed)
    else:
        noise_shape = _process_noise_dim(shape, noise_dims)
        # auto select broadcast shape
        broadcast = [i for i, j in enumerate(noise_shape) if j == 1]
        if len(broadcast) > 0:
            x = x * K.addbroadcast(
                K.random_binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype, seed=seed),
                *broadcast)
        else:
            x = x * K.random_binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype, seed=seed)
    if rescale:
        x /= retain_prob
    if isinstance(input_shape, (tuple, list)):
        K.add_shape(x, input_shape)
    return x


def apply_noise(x, sigma=0.075, noise_dims=None, noise_type='gaussian', seed=None):
    """
    Parameters
    ----------
    x: A tensor.
    sigma : float or tensor scalar
        Standard deviation of added Gaussian noise
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """
    input_shape = K.shape(x)
    noise_type = noise_type.lower()
    # ====== not a training variable NO dropout ====== #
    if not K.is_training(x):
        return x
    # ====== applying noise ====== #
    shape = K.shape(x, none=False)
    noise_shape = (shape if noise_dims is None
                   else _process_noise_dim(shape, noise_dims))
    if 'normal' in noise_type or 'gaussian' in noise_type:
        noise = K.random_normal(shape=noise_shape, mean=0.0, std=sigma, dtype=x.dtype, seed=seed)
    elif 'uniform' in noise_type:
        noise = K.random_uniform(shape=noise_shape, low=-sigma, high=sigma, dtype=x.dtype, seed=seed)
        # no idea why uniform does not give any broadcastable dimensions
        if noise_dims is not None:
            broadcastable = [i for i, j in enumerate(noise_shape) if j == 1]
            if len(broadcastable) > 0:
                noise = K.addbroadcast(noise, *broadcastable)
    x = x + noise
    if isinstance(input_shape, (tuple, list)):
        K.add_shape(x, input_shape)
    return x


# ===========================================================================
# Variational OPERATIONS
# ===========================================================================
def log_prob_bernoulli(p_true, p_approx, mask=None):
    """ Compute log probability of some binary variables with probabilities
    given by p_true, for probability estimates given by p_approx. We'll
    compute joint log probabilities over row-wise groups.
    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    if mask is None:
        mask = T.ones((1, p_approx.shape[1]))
    log_prob_1 = p_true * T.log(p_approx)
    log_prob_0 = (1.0 - p_true) * T.log(1.0 - p_approx)
    log_prob_01 = log_prob_1 + log_prob_0
    row_log_probs = T.sum((log_prob_01 * mask), axis=1, keepdims=True)
    return row_log_probs

#logpxz = -0.5*np.log(2 * np.pi) - log_sigma_decoder - (0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)


def log_prob_gaussian(mu_true, mu_approx, les_sigmas=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and standard deviations given by les_sigmas.
    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    ind_log_probs = C - T.log(T.abs_(les_sigmas)) - \
    ((mu_true - mu_approx)**2.0 / (2.0 * les_sigmas**2.0))
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return row_log_probs


def log_prob_gaussian2(mu_true, mu_approx, log_vars=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and log variances given by les_logvars.
    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    ind_log_probs = C - (0.5 * log_vars) - \
    ((mu_true - mu_approx)**2.0 / (2.0 * T.exp(log_vars)))
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return row_log_probs
