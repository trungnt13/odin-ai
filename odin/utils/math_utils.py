from __future__ import print_function, division, absolute_import

import math
import numpy as np


# ===========================================================================
# Interpolation
# ===========================================================================
def _power_interp(a, power):
    return np.where(a <= 0.5,
                    np.power(a * 2, power) / 2,
                    np.power((a - 1) * 2, power) / (-2 if power % 2 == 0 else 2) + 1)


def _exp_interp(a, value, power):
    min = np.power(value, -float(power))
    scale = 1 / (1 - min)
    return np.where(a <= 0.5,
        (np.power(value, power * (a * 2 - 1)) - min) * scale / 2,
        (2 - (np.power(value, -power * (a * 2 - 1)) - min) * scale) / 2)


def _expIn_interp(a, value, power):
    min = np.power(value, -float(power))
    scale = 1 / (1 - min)
    return (np.power(value, power * (a - 1)) - min) * scale


def _expOut_interp(a, value, power):
    min = np.power(value, -float(power))
    scale = 1 / (1 - min)
    return 1 - (np.power(value, -power * a) - min) * scale


def _0_1_array(n):
    return np.linspace(start=0., stop=1., num=int(n))


class interp(object):
    """ Original code, libgdx
    https://github.com/libgdx/libgdx/wiki/Interpolation

    Return an array of interpolated values within given range: [start, end]
    """
    @staticmethod
    def circle(n):
        a = _0_1_array(n)
        a1 = (a - 1) * 2
        return np.where(a <= 0.5,
                        (1 - np.sqrt(1 - np.power(2 * a, 2.))) / 2,
                        (np.sqrt(1 - np.power(a1, 2.) + 1) / 2))

    @staticmethod
    def pow(n, power):
        return _power_interp(_0_1_array(n), power)

    @staticmethod
    def powIn(n, power):
        return np.power(_0_1_array(n), power)

    @staticmethod
    def powOut(n, power):
        return np.power(_0_1_array(n) - 1, power) * (-1 if power % 2 == 0 else 1) + 1

    @staticmethod
    def exp(n, power):
        return _exp_interp(_0_1_array(n), value=2, power=power)

    @staticmethod
    def expIn(n, power):
        return _expIn_interp(_0_1_array(n), value=2, power=power)

    @staticmethod
    def expOut(n, power):
        return _expOut_interp(_0_1_array(n), value=2, power=power)

    @staticmethod
    def swing(n, scale=1.5):
        scale = scale * 2
        a = _0_1_array(n)
        a1 = (a - 1) * 2
        a2 = a * 2
        return np.where(a <= 0.5,
                        np.power(a2, 2.) * ((scale + 1) * a2 - scale) / 2,
                        np.power(a1, 2.) * ((scale + 1) * a1 + scale) / 2 + 1)

    @staticmethod
    def swingIn(n, scale=2):
        a = _0_1_array(n)
        return a * a * ((scale + 1) * a - scale)

    @staticmethod
    def swingOut(n, scale=2):
        a = _0_1_array(n) - 1.
        return a * a * ((scale + 1) * a + scale) + 1
