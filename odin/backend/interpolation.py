# Original implementation: Libgdx
# License: https://github.com/libgdx/libgdx/blob/master/LICENSE
from __future__ import absolute_import, division, print_function

import re
from enum import Enum, auto

import numpy as np
import tensorflow as tf

pattern = re.compile(r'^([a-z]+)(\d*)([a-zA-Z]*)')


def cbrt(x):
  return tf.pow(x, 1 / 3)


class Interpolation(Enum):
  r"""
  """

  linear = auto()
  smooth = auto()
  smooth2 = auto()
  fade = auto()
  # power
  pow2 = auto()
  pow2In = auto()
  pow2Out = auto()
  pow2InInverse = auto()
  pow2OutInverse = auto()
  pow3 = auto()
  pow3In = auto()
  pow3Out = auto()
  pow3InInverse = auto()
  pow3OutInverse = auto()
  pow4 = auto()
  pow4In = auto()
  pow4Out = auto()
  pow5 = auto()
  pow5In = auto()
  pow5Out = auto()
  # sine
  sine = auto()
  sineIn = auto()
  sineOut = auto()
  # circle
  circle = auto()
  circleIn = auto()
  circleOut = auto()
  # exp
  exp5 = auto()
  exp5In = auto()
  exp5Out = auto()
  exp10 = auto()
  exp10In = auto()
  exp10Out = auto()
  # swing
  swing = auto()
  swingIn = auto()
  swingOut = auto()
  # elastic
  elastic = auto()
  elasticIn = auto()
  elasticOut = auto()

  def __call__(self, a, norm=None, cyclical=False, delay=0., vmin=0., vmax=1.):
    return self.apply(a, norm, cyclical, vmin, vmax)

  def apply(self, a, norm=None, cyclical=False, delay=0., vmin=0., vmax=1.):
    r"""
    Arguments:
      a : Scalar.
      norm : Scalar (optional)
      cyclical : Boolean. Enable cyclical scheduling, `norm` determines the
        cycle periodic.
      delay : Scalar. The amount of delay before each cycle reseted.
      vmin : Scalar (default: 0). Minimum value for the interpolation output,
        the return range is [vmin, vmax]
      vmax : Scalar (default: 1). Maximum value for the interpolation output,
        the return range is [vmin, vmax]
    """
    if norm is not None:
      a = tf.maximum(tf.cast(a, 'float32'), 1e-8)
      if cyclical:
        a = a % (norm + delay) + 1
        a = tf.minimum(a, norm)
      a = a / norm
    a = tf.maximum(0., tf.minimum(a, 1.))

    name = str(self).split('.')[-1]
    _, name, power, mode, _ = pattern.split(name)
    power = float(power) if len(power) > 0 else 1.

    isinverse = 'Inverse' in mode
    mode = mode.replace('Inverse', '')
    isin = 'In' in mode
    isout = 'Out' in mode

    if name == 'linear':
      pass
    elif 'smooth' == name[:6]:
      a = a * a * (3 - 2 * a)
      if '2' == name[-1]:
        a = a * a * (3 - 2 * a)
    elif name in ('fade', 'smoother'):
      a = a * a * a * (a * (a * 6 - 15) + 10)
    elif 'pow' == name[:3]:
      if isin:
        a = (tf.sqrt(a) if power == 2 else cbrt(a)) \
          if isinverse else \
            tf.pow(a, power)
      elif isout:
        a = (1 - tf.sqrt(-(a - 1)) if power == 2 else 1 - cbrt(-(a - 1))) \
          if isinverse else \
            tf.pow(a - 1, power) * (-1 if power % 2 == 0 else 1) + 1
      else:
        a = tf.where(
            a <= 0.5,
            tf.pow(a * 2, power) / 2,
            tf.pow((a - 1) * 2, power) / (-2 if power % 2 == 0 else 2) + 1)
    elif 'sine' == name[:4]:
      if isin:
        a = 1 - tf.cos(a * np.pi / 2)
      elif isout:
        a = tf.sin(a * np.pi / 2)
      else:
        a = (1 - tf.cos(a * np.pi)) / 2
    elif 'circle' == name[:6]:
      if isin:
        a = 1 - tf.sqrt(1 - a * a)
      elif isout:
        a = tf.sqrt(1 - (a - 1)**2)
      else:
        a = tf.where(
          a <= 0.5, \
          (1 - tf.sqrt(1 - (a * 2)**2)) / 2, \
          (tf.sqrt(1 - ((a - 1) * 2)**2) + 1) / 2)
    elif 'exp' == name[:3]:
      base = 2.
      min_val = tf.pow(base, -power)
      scale = 1 / (1 - min_val)
      if isin:
        a = (tf.pow(base, power * (a - 1)) - min_val) * scale
      elif isout:
        a = 1 - (tf.pow(base, -power * a) - min_val) * scale
      else:
        a = tf.where(
            a <= 0.5, \
            (tf.pow(base, power * (a * 2 - 1)) - min_val) * scale / 2, \
            (2 - (tf.pow(base, -power * (a * 2 - 1)) - min_val) * scale) / 2)
    elif 'swing' == name[:5]:
      if isin:
        scale = 2
        a = a * a * ((scale + 1) * a - scale)
      elif isout:
        scale = 2
        a = a - 1
        a = a * a * ((scale + 1) * a + scale) + 1
      else:
        scale = 3
        a = tf.where(
          a <= 0.5, \
          (a * 2) ** 2 * ((scale + 1) * a * 2 - scale) / 2, \
          ((a - 1) * 2) ** 2 * ((scale + 1) * ((a - 1) * 2) + scale) / 2 + 1 \
        )
    elif 'elastic' == name[:7]:
      base = 2.
      power = 10.
      scale = 1.
      bounces = 6 if isin else 7
      bounces *= np.pi * (1 if bounces % 2 == 0 else -1)
      if isin:
        a = tf.where(
          a >= 0.99, \
          tf.ones_like(a), \
          tf.pow(base, power * (a - 1)) * tf.sin(a * bounces) * scale
        )
      elif isout:
        a = tf.where(
          a == 0, \
          tf.zeros_like(a), \
          1 - tf.pow(base, power * ((1 - a) - 1)) * tf.sin((1 - a) * bounces) * scale
        )
      else:
        a = tf.where(
          a <= 0.5, \
          tf.pow(base, power * (a * 2 - 1)) * tf.sin(a * 2 * bounces) * scale / 2, \
          1 - tf.pow(base, power * ((1 - a) * 2 - 1)) * tf.sin((1 - a) * 2 * bounces) * scale / 2
        )

    return (vmax - vmin) * a + vmin
