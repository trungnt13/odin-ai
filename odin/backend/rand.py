from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from odin.utils import uuid, as_tuple
from odin.config import get_rng, CONFIG, randint

from .helpers import is_training
from .roles import add_role, Weight, Bias, Parameter

floatX = CONFIG.floatX


def random_binomial(shape, p, dtype=floatX, seed=None, name="RandomBinomal"):
    with tf.variable_scope(name):
        return tf.where(
            tf.random_uniform(shape, minval=0., maxval=1., dtype=dtype, seed=seed) <= p,
            tf.ones(shape, dtype=dtype),
            tf.zeros(shape, dtype=dtype))


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
    => Noise mask: (None, 10, 1)
    """
    if dims is None:
        return input_shape
    ndims = input_shape.shape[0].value
    dims = [i % ndims for i in as_tuple(dims, t=int)]
    # ====== get noise shape ====== #
    return tuple([1 if i in dims else input_shape[i]
                  for i in range(ndims)])


def apply_dropout(x, level=0.5, noise_dims=None, noise_type='uniform',
                  rescale=True, name="ApplyDropout"):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.


    Parameters
    ----------
    x: A tensor.
        input tensor
    level: float(0.-1.)
        probability dropout values in given tensor
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    seed: random seed or `tensor.rng`
        random generator from tensor class

    References
    ----------
    [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

    Note
    ----
    This function only apply noise on Variable when training is enable
    """
    shape = tf.shape(x)
    retain_prob = 1. - level
    # ====== not a training variable NO dropout ====== #
    if 'normal' in noise_type or 'gaussian' in noise_type:
        randfunc = lambda shape: tf.random_normal(shape=shape,
            mean=1.0, stddev=np.sqrt((1.0 - retain_prob) / retain_prob),
            dtype=x.dtype.base_dtype, seed=randint())
    elif 'uniform' in noise_type:
        randfunc = lambda shape: random_binomial(shape=shape,
            p=retain_prob, dtype=x.dtype.base_dtype, seed=randint())
    else:
        raise ValueError('No support for noise_type=' + noise_type)

    # ====== Dropout ====== #
    def training_fn():
        noise_shape = shape if noise_dims is None else \
            _process_noise_dim(shape, noise_dims)
        y = x * randfunc(shape=noise_shape)
        if rescale:
            y /= retain_prob
        return y

    def inference_fn():
        return x
    with tf.variable_scope(name):
        return tf.cond(is_training(), training_fn, inference_fn)


def apply_noise(x, level=0.075, noise_dims=None, noise_type='gaussian',
                name="ApplyNoise"):
    """
    Parameters
    ----------
    x: A tensor.
    level : float or tensor scalar
        Standard deviation of added Gaussian noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    seed: random seed or `tensor.rng`
        random generator from tensor class

    Note
    ----
    This function only apply noise on Variable when training is enable
    """
    noise_type = noise_type.lower()
    shape = tf.shape(x)

    # ====== applying noise ====== #
    def training_fn():
        noise_shape = (shape if noise_dims is None
                       else _process_noise_dim(shape, noise_dims))
        if 'normal' in noise_type or 'gaussian' in noise_type:
            noise = tf.random_normal(shape=noise_shape,
                mean=0.0, stddev=level, dtype=x.dtype.base_dtype, seed=randint())
        elif 'uniform' in noise_type:
            noise = tf.random_uniform(shape=noise_shape,
                minval=-level, maxval=level,
                dtype=x.dtype.base_dtype, seed=randint())
        else:
            raise ValueError('No support for noise_type=' + noise_type)
        return x + noise

    # ====== inference_fn ====== #
    def inference_fn():
        return x
    with tf.variable_scope(name):
        return tf.cond(is_training(), training_fn, inference_fn)


# ===========================================================================
# Special random algorithm for weights initialization
# ===========================================================================
def normal(shape, mean=0., std=1.):
    return np.cast[floatX](
        get_rng().normal(mean, std, size=shape))


def uniform(shape, range=0.05):
    if isinstance(range, (int, float, long)):
        range = (-abs(range), abs(range))
    return np.cast[floatX](
        get_rng().uniform(low=range[0], high=range[1], size=shape))


class constant(object):

    def __init__(self, val):
        super(constant, self).__init__()
        self.__name__ = 'constant'
        self.val = val

    def __call__(self, shape):
        return np.cast[floatX](np.zeros(shape) + self.val)


def symmetric_uniform(shape, range=0.01, std=None, mean=0.0):
    if std is not None:
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
    else:
        try:
            a, b = range  # range is a tuple
        except TypeError:
            a, b = -range, range  # range is a number
    return np.cast[floatX](
        get_rng().uniform(low=a, high=b, size=shape))


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
    return np.cast[floatX](
        get_rng().uniform(low=a, high=b, size=orig_shape))


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
    return np.cast[floatX](
        get_rng().normal(0.0, std, size=orig_shape))


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
    return np.cast[floatX](
        get_rng().normal(0.0, std, size=shape))


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
    return np.cast[floatX](
        get_rng().uniform(low=a, high=b, size=shape))


def orthogonal(shape, gain=1.0):
    if gain == 'relu':
        gain = np.sqrt(2)

    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported, but "
                           "given shape:%s" % str(shape))

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = get_rng().normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.cast[floatX](gain * q)


# ===========================================================================
# Fast initialization
# ===========================================================================
def rnn(input_dim, hidden_dim,
        W_init=glorot_uniform, b_init=constant(0.),
        bidirectional=False, one_vector=False,
        return_variable=True, name=None):
    """ Fast initalize all Standard RNN weights

    Parameters
    ----------
    one_vector: bool
        if True, all the weights are flatten and concatenated into 1 big vector
    return_variable: bool
        if False, only return the numpy array
    bidirectional: bool
        if True, return parameters for both forward and backward RNN

    Return
    ------
    [W_i, b_wi, R_h, b_wh]

    """
    if name is None: name = uuid()

    def init():
        W_i = W_init((input_dim, hidden_dim))
        b_wi = b_init((hidden_dim))
        R_h = W_init((hidden_dim, hidden_dim))
        b_wh = b_init((hidden_dim))
        return [W_i, b_wi, R_h, b_wh]
    params = init() + init() if bidirectional else init()
    roles = [Weight, Bias]
    if one_vector:
        params = [np.concatenate([p.flatten() for p in params])]
        roles = [Parameter]
    # names
    if one_vector:
        names = [name + '_rnn']
    else:
        names = ["_W_i", "_b_wi", "_R_h", "_b_wh"]
        if bidirectional:
            names = [i + '_fw' for i in names] + [i + '_bw' for i in names]
        names = [name + i for i in names]
    # create variable or not
    if return_variable:
        params = [tf.Variable(p, dtype=floatX, name=n)
                  for p, n in zip(params, names)]
        for i, p in enumerate(params):
            add_role(p, roles[i % 2])
    return params if len(params) > 1 else params[0]


def lstm(input_dim, hidden_dim,
        W_init=glorot_uniform, b_init=constant(0.),
        bidirectional=False, one_vector=False,
        return_variable=True, name=None):
    """ Fast initalize all Standard LSTM weights (without peephole connection)

    Parameters
    ----------
    one_vector: bool
        if True, all the weights are flatten and concatenated into 1 big vector
    return_variable: bool
        if False, only return the numpy array
    bidirectional: bool
        if True, return parameters for both forward and backward RNN

    Return
    ------
    [W_i, b_wi, W_f, b_wf, W_c, b_wc, W_o, b_wo,
     R_i, b_ri, R_f, b_rf, R_c, b_rc, R_o, b_ro]

    """
    if name is None: name = uuid()

    def init():
        # input to hidden
        W_i = W_init((input_dim, hidden_dim))
        b_wi = b_init((hidden_dim))
        W_f = W_init((input_dim, hidden_dim))
        b_wf = b_init((hidden_dim))
        W_c = W_init((input_dim, hidden_dim))
        b_wc = b_init((hidden_dim))
        W_o = W_init((input_dim, hidden_dim))
        b_wo = b_init((hidden_dim))
        # hidden to hidden
        R_i = W_init((hidden_dim, hidden_dim))
        b_ri = b_init((hidden_dim))
        R_f = W_init((hidden_dim, hidden_dim))
        b_rf = b_init((hidden_dim))
        R_c = W_init((hidden_dim, hidden_dim))
        b_rc = b_init((hidden_dim))
        R_o = W_init((hidden_dim, hidden_dim))
        b_ro = b_init((hidden_dim))
        return [W_i, b_wi, W_f, b_wf, W_c, b_wc, W_o, b_wo,
              R_i, b_ri, R_f, b_rf, R_c, b_rc, R_o, b_ro]
    params = init() + init() if bidirectional else init()
    roles = [Weight, Bias]
    if one_vector:
        params = [np.concatenate([p.flatten() for p in params])]
        roles = [Parameter]
    # names
    if one_vector:
        names = [name + '_lstm']
    else:
        names = ["_W_i", "_b_wi", "_W_f", "_b_wf", "_W_c", "_b_wc", "_W_o", "_b_wo",
                 "_R_i", "_b_ri", "_R_f", "_b_rf", "_R_c", "_b_rc", "_R_o", "_b_ro"]
        if bidirectional:
            names = [i + '_fw' for i in names] + [i + '_bw' for i in names]
        names = [name + i for i in names]
    # create variable or not
    if return_variable:
        params = [tf.Variable(p, dtype=floatX, name=n)
                  for p, n in zip(params, names)]
        for i, p in enumerate(params):
            add_role(p, roles[i % 2])
    return params if len(params) > 1 else params[0]


def gru(input_dim, hidden_dim,
        W_init=glorot_uniform, b_init=constant(0.),
        bidirectional=False, one_vector=False,
        return_variable=True, name=None):
    """ Fast initalize all Standard GRU weights

    Parameters
    ----------
    one_vector: bool
        if True, all the weights are flatten and concatenated into 1 big vector
    return_variable: bool
        if False, only return the numpy array
    bidirectional: bool
        if True, return parameters for both forward and backward RNN

    Return
    ------
    [W_r, b_wr, W_i, b_wi,
     W_h, b_wh, R_r, b_rr,
     R_i, b_ru, R_h, b_rh]
    """
    if name is None: name = uuid()

    def init():
        W_r = W_init((input_dim, hidden_dim))
        b_wr = b_init((hidden_dim))
        W_i = W_init((input_dim, hidden_dim))
        b_wi = b_init((hidden_dim))
        W_h = W_init((input_dim, hidden_dim))
        b_wh = b_init((hidden_dim))
        R_r = W_init((hidden_dim, hidden_dim))
        b_rr = b_init((hidden_dim))
        R_i = W_init((hidden_dim, hidden_dim))
        b_ru = b_init((hidden_dim))
        R_h = W_init((hidden_dim, hidden_dim))
        b_rh = b_init((hidden_dim))
        return [W_r, b_wr, W_i, b_wi, W_h, b_wh,
                R_r, b_rr, R_i, b_ru, R_h, b_rh]
    params = init() + init() if bidirectional else init()
    roles = [Weight, Bias]
    if one_vector:
        params = [np.concatenate([p.flatten() for p in params])]
        roles = [Parameter]
    # names
    if one_vector:
        names = [name + '_gru']
    else:
        names = ["_W_r", "_b_wr", "_W_i", "_b_wi", "_W_h", "_b_wh",
                 "_R_r", "_b_rr", "_R_i", "_b_ru", "_R_h", "_b_rh"]
        if bidirectional:
            names = [i + '_fw' for i in names] + [i + '_bw' for i in names]
        names = [name + i for i in names]
    # create variable or not
    if return_variable:
        params = [tf.Variable(p, dtype=floatX, name=n)
                  for p, n in zip(params, names)]
        for i, p in enumerate(params):
            add_role(p, roles[i % 2])
    return params if len(params) > 1 else params[0]
