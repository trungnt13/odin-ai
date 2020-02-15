# Original implementation: Libgdx
# License: https://github.com/libgdx/libgdx/blob/master/LICENSE
from __future__ import absolute_import, division, print_function

import inspect

import numpy as np
import tensorflow as tf


def cbrt(x):
  return tf.pow(x, 1 / 3)


def get(name=None):
  all_interpolation = {}
  for key, val in globals().items():
    if inspect.isclass(val) and issubclass(
        val, Interpolation) and val is not Interpolation:
      all_interpolation[key] = val
  if name is None:
    return [i[-1] for i in sorted(all_interpolation.items())]
  return all_interpolation[name]


# ===========================================================================
# Base class
# ===========================================================================
class Interpolation(object):
  r""" Interpolation algorithm

  Arguments:
    vmin : Scalar (default: 0). Minimum value for the interpolation output,
      the return range is [vmin, vmax]
    vmax : Scalar (default: 1). Maximum value for the interpolation output,
      the return range is [vmin, vmax]
    norm : Scalar (optional). Normalization constant for the input value,
      the repeat cycle in case of cyclical scheduling.
    cyclical : Boolean. Enable cyclical scheduling, `norm` determines the
      cycle periodic.
    delayIn : Scalar. The amount of delay at the beginning of each cycle.
    delayOut : Scalar. The amount of delay at the end of each cycle.
  """

  def __init__(self,
               vmin=0.,
               vmax=1.,
               norm=1,
               cyclical=False,
               delayIn=0,
               delayOut=0):
    self.vmin = vmin
    self.vmax = vmax
    self.norm = norm
    self.cyclical = cyclical
    self.delayIn = max(delayIn, 0)
    self.delayOut = max(delayOut, 0)

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "<%s(%.2f,%.2f,%d) cyclical:%s delay:(%d,%d)>" % \
      (self.__class__.__name__, self.vmin, self.vmax,
       self.norm, self.cyclical, self.delayIn, self.delayOut)

  def __call__(self, a):
    return self.apply(a)

  def apply(self, a):
    r""" Applying the interpolation algorithm to return value within the range
      `[vmin, vmax]`

    Arguments:
      a : A Scalar or an Array. Expect input value in range [0, 1]
    """
    a = tf.maximum(tf.cast(a, 'float32'), 1e-8)
    if self.cyclical:
      a = a % (self.delayIn + self.norm + self.delayOut) + 1
      a = a - self.delayIn
      a = tf.maximum(tf.minimum(a, self.norm), 0.)
    a = a / self.norm
    a = tf.maximum(0., tf.minimum(a, 1.))
    return (self.vmax - self.vmin) * self._alpha(a) + self.vmin

  def _alpha(self, a):
    raise NotImplementedError()


# ===========================================================================
# Interpolation methods
# ===========================================================================
class const(Interpolation):

  def apply(self, a):
    a = a + 1e-8
    return a / a * self.vmax


class linear(Interpolation):

  def _alpha(self, a):
    return a


class smooth(Interpolation):

  def _alpha(self, a):
    return a * a * (3 - 2 * a)


class smooth2(Interpolation):

  def _alpha(self, a):
    return a * a * (3 - 2 * a)


class fade(Interpolation):

  def _alpha(self, a):
    return a * a * a * (a * (a * 6 - 15) + 10)


smoother = fade


# ===========================================================================
# Power
# ===========================================================================
class power(Interpolation):

  def __init__(self,
               power=2.,
               inverse=False,
               vmin=0.,
               vmax=1.,
               norm=1,
               cyclical=False,
               delayIn=0,
               delayOut=0):
    super().__init__(vmin=vmin,
                     vmax=vmax,
                     norm=norm,
                     cyclical=cyclical,
                     delayIn=delayIn,
                     delayOut=delayOut)
    self.power = power
    self.inverse = bool(inverse)

  def _alpha(self, a):
    return tf.where(
        a <= 0.5,
        tf.pow(a * 2, self.power) / 2,
        tf.pow((a - 1) * 2, self.power) / ((self.power % 2 - 0.5) * 4) + 1)


class powerIn(power):

  def _alpha(self, a):
    power = self.power
    if self.inverse:
      return tf.pow(a, 1. / power)
    return tf.pow(a, power)


class powerOut(power):

  def _alpha(self, a):
    power = self.power
    if self.inverse:
      return 1 - tf.pow(-(a - 1), 1. / power)
    return tf.pow(a - 1, power) * (power % 2 - 0.5) * 2 + 1


# ===========================================================================
# Sine and circle
# ===========================================================================
class sine(Interpolation):

  def _alpha(self, a):
    return (1 - tf.cos(a * np.pi)) / 2


class sineIn(Interpolation):

  def _alpha(self, a):
    return 1 - tf.cos(a * np.pi / 2)


class sineOut(Interpolation):

  def _alpha(self, a):
    return tf.sin(a * np.pi / 2)


class circle(Interpolation):

  def _alpha(self, a):
    return tf.where(a <= 0.5, (1 - tf.sqrt(1 - (a * 2)**2)) / 2,
                    (tf.sqrt(1 - ((a - 1) * 2)**2) + 1) / 2)


class circleIn(Interpolation):

  def _alpha(self, a):
    return 1 - tf.sqrt(1 - a * a)


class circleOut(Interpolation):

  def _alpha(self, a):
    return tf.sqrt(1 - tf.pow((a - 1), 2))


# ===========================================================================
# Swing
# ===========================================================================
class swing(Interpolation):

  def __init__(self,
               scale=3,
               vmin=0.,
               vmax=1.,
               norm=1,
               cyclical=False,
               delayIn=0,
               delayOut=0):
    super().__init__(vmin=vmin,
                     vmax=vmax,
                     norm=norm,
                     cyclical=cyclical,
                     delayIn=delayIn,
                     delayOut=delayOut)
    self.scale = scale

  def _alpha(self, a):
    scale = self.scale
    return tf.where(
          a <= 0.5, \
          (a * 2) ** 2 * ((scale + 1) * a * 2 - scale) / 2, \
          ((a - 1) * 2) ** 2 * ((scale + 1) * ((a - 1) * 2) + scale) / 2 + 1 \
        )


class swingIn(swing):

  def __init__(self,
               scale=2,
               vmin=0.,
               vmax=1.,
               norm=1,
               cyclical=False,
               delayIn=0,
               delayOut=0):
    super().__init__(scale=scale,
                     vmin=vmin,
                     vmax=vmax,
                     norm=norm,
                     cyclical=cyclical,
                     delayIn=delayIn,
                     delayOut=delayOut)

  def _alpha(self, a):
    scale = self.scale
    return a * a * ((scale + 1) * a - scale)


class swingOut(swingIn):

  def _alpha(self, a):
    scale = self.scale
    a = a - 1
    return a * a * ((scale + 1) * a + scale) + 1


# ===========================================================================
# Exponent
# ===========================================================================
class exp(Interpolation):

  def __init__(self,
               base=2.,
               power=5.,
               vmin=0.,
               vmax=1.,
               norm=1,
               cyclical=False,
               delayIn=0,
               delayOut=0):
    super().__init__(vmin=vmin,
                     vmax=vmax,
                     norm=norm,
                     cyclical=cyclical,
                     delayIn=delayIn,
                     delayOut=delayOut)
    self.base = base
    self.power = power
    self.min_val = tf.pow(base, -power)
    self.scale = 1 / (1 - self.min_val)

  def _alpha(self, a):
    base = self.base
    power = self.power
    min_val = self.min_val
    scale = self.scale
    return tf.where(
            a <= 0.5, \
            (tf.pow(base, power * (a * 2 - 1)) - min_val) * scale / 2, \
            (2 - (tf.pow(base, -power * (a * 2 - 1)) - min_val) * scale) / 2)


class expIn(exp):

  def _alpha(self, a):
    base = self.base
    power = self.power
    min_val = self.min_val
    scale = self.scale
    return (tf.pow(base, power * (a - 1)) - min_val) * scale


class expOut(exp):

  def _alpha(self, a):
    base = self.base
    power = self.power
    min_val = self.min_val
    scale = self.scale
    return 1 - (tf.pow(base, -power * a) - min_val) * scale


# ===========================================================================
# Elastic
# ===========================================================================
class elastic(Interpolation):

  def __init__(self,
               base=2.,
               power=10.,
               scale=1.,
               bounces=7.,
               vmin=0.,
               vmax=1.,
               norm=1,
               cyclical=False,
               delayIn=0,
               delayOut=0):
    super().__init__(vmin=vmin,
                     vmax=vmax,
                     norm=norm,
                     cyclical=cyclical,
                     delayIn=delayIn,
                     delayOut=delayOut)
    self.base = base
    self.power = power
    self.scale = scale
    self.bounces = bounces
    self.bounces *= np.pi * (1 if bounces % 2 == 0 else -1)

  def _alpha(self, a):
    base = self.base
    power = self.power
    scale = self.scale
    bounces = self.bounces
    return tf.where(
      a <= 0.5, \
      tf.pow(base, power * (a * 2 - 1)) * tf.sin(a * 2 * bounces) * scale / 2, \
      1 - tf.pow(base, power * ((1 - a) * 2 - 1)) * tf.sin((1 - a) * 2 * bounces) * scale / 2
    )


class elasticIn(elastic):

  def _alpha(self, a):
    base = self.base
    power = self.power
    scale = self.scale
    bounces = self.bounces
    return tf.where(
      a >= 0.99, \
      tf.ones_like(a), \
      tf.pow(base, power * (a - 1)) * tf.sin(a * bounces) * scale
    )


class elasticOut(elastic):

  def _alpha(self, a):
    base = self.base
    power = self.power
    scale = self.scale
    bounces = self.bounces
    return tf.where(
      a == 0, \
      tf.zeros_like(a), \
      1 - tf.pow(base, power * ((1 - a) - 1)) * tf.sin((1 - a) * bounces) * scale
    )
