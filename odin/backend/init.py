from __future__ import print_function, division, absolute_import

import numpy as np

from odin.config import get_rng, CONFIG
from odin.utils import uuid
from .basic_ops import variable

FLOATX = CONFIG.floatX


# ===========================================================================
# Special random algorithm for weights initialization
# ===========================================================================
def normal(shape, mean=0., std=1.):
    return np.cast[FLOATX](
        get_rng().normal(mean, std, size=shape))


def uniform(shape, range=0.05):
    if isinstance(range, (int, float, long)):
        range = (-abs(range), abs(range))
    return np.cast[FLOATX](
        get_rng().uniform(low=range[0], high=range[1], size=shape))


class constant(object):

    def __init__(self, val):
        super(constant, self).__init__()
        self.__name__ = 'constant'
        self.val = val

    def __call__(self, shape):
        return np.cast[FLOATX](np.zeros(shape) + self.val)


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
    return np.cast[FLOATX](
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
    return np.cast[FLOATX](
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
    return np.cast[FLOATX](
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
    return np.cast[FLOATX](
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
    return np.cast[FLOATX](gain * q)


# ===========================================================================
# Fast initialization
# ===========================================================================
from odin.basic import add_role, Weight, Bias, Parameter


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
        params = [variable(p, name=n) for p, n in zip(params, names)]
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
        params = [variable(p, name=n) for p, n in zip(params, names)]
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
        params = [variable(p, name=n) for p, n in zip(params, names)]
        for i, p in enumerate(params):
            add_role(p, roles[i % 2])
    return params if len(params) > 1 else params[0]
