# Original implementation: Libgdx
# License: https://github.com/libgdx/libgdx/blob/master/LICENSE
from __future__ import absolute_import, division, print_function

import re
from enum import Enum, auto

import numpy as np

pattern = re.compile(r'^([a-z]+)(\d*)([a-zA-Z]*)')


class Interpolation(Enum):
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

  def __call__(self, a, vmin=0., vmax=1.):
    name = str(self).split('.')[-1]
    _, name, power, mode, _ = pattern.split(name)
    power = int(power) if len(power) > 0 else 1

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
        a = (np.sqrt(a) if power == 2 else np.cbrt(a)) \
          if isinverse else \
            np.power(a, power)
      elif isout:
        a = (1 - np.sqrt(-(a - 1)) if power == 2 else 1 - np.cbrt(-(a - 1))) \
          if isinverse else \
            np.power(a - 1, power) * (-1 if power % 2 == 0 else 1) + 1
      else:
        a = np.where(
            a <= 0.5,
            np.power(a * 2, power) / 2,
            np.power((a - 1) * 2, power) / (-2 if power % 2 == 0 else 2) + 1)
    elif 'sine' == name[:4]:
      if isin:
        a = 1 - np.cos(a * np.pi / 2)
      elif isout:
        a = np.sin(a * np.pi / 2)
      else:
        a = (1 - np.cos(a * np.pi)) / 2
    elif 'circle' == name[:6]:
      if isin:
        a = 1 - np.sqrt(1 - a * a)
      elif isout:
        a = np.sqrt(1 - (a - 1)**2)
      else:
        a = np.where(
          a <= 0.5, \
          (1 - np.sqrt(1 - (a * 2)**2)) / 2, \
          (np.sqrt(1 - ((a - 1) * 2)**2) + 1) / 2)
    elif 'exp' == name[:3]:
      base = 2.
      min_val = np.power(base, -power)
      scale = 1 / (1 - min_val)
      if isin:
        a = (np.power(base, power * (a - 1)) - min_val) * scale
      elif isout:
        a = 1 - (np.power(base, -power * a) - min_val) * scale
      else:
        a = np.where(
            a <= 0.5, \
            (np.power(base, power * (a * 2 - 1)) - min_val) * scale / 2, \
            (2 - (np.power(base, -power * (a * 2 - 1)) - min_val) * scale) / 2)
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
        a = np.where(
          a <= 0.5, \
          (a * 2) ** 2 * ((scale + 1) * a * 2 - scale) / 2, \
          ((a - 1) * 2) ** 2 * ((scale + 1) * ((a - 1) * 2) + scale) / 2 + 1 \
        )
    elif 'elastic' == name[:7]:
      base = 2
      power = 10
      scale = 1
      bounces = 6 if isin else 7
      bounces *= np.pi * (1 if bounces % 2 == 0 else -1)
      if isin:
        a = np.where(
          a >= 0.99, \
          np.ones_like(a), \
          np.power(base, power * (a - 1)) * np.sin(a * bounces) * scale
        )
      elif isout:
        a = np.where(
          a == 0, \
          np.zeros_like(a), \
          1 - np.power(base, power * ((1 - a) - 1)) * np.sin((1 - a) * bounces) * scale
        )
      else:
        a = np.where(
          a <= 0.5, \
          np.power(base, power * (a * 2 - 1)) * np.sin(a * 2 * bounces) * scale / 2, \
          1 - np.power(base, power * ((1 - a) * 2 - 1)) * np.sin((1 - a) * 2 * bounces) * scale / 2
        )

    return (vmax - vmin) * a + vmin
