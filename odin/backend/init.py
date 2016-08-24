from __future__ import print_function, division, absolute_import

import numpy as np

from odin.config import RNG_GENERATOR, autoconfig
FLOATX = autoconfig.floatX


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
