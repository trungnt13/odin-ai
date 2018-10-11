from __future__ import print_function, division, absolute_import

import math
import numpy as np

# ===========================================================================
# Distance measurement
# ===========================================================================
def _LevenshteinDistance(s1, s2):
  ''' Implementation of the wikipedia algorithm, optimized for memory
  Reference: http://rosettacode.org/wiki/Levenshtein_distance#Python
  '''
  if len(s1) > len(s2):
    s1, s2 = s2, s1
  distances = range(len(s1) + 1)
  for index2, char2 in enumerate(s2):
    newDistances = [index2 + 1]
    for index1, char1 in enumerate(s1):
      if char1 == char2:
        newDistances.append(distances[index1])
      else:
        newDistances.append(1 + min((distances[index1],
                                     distances[index1 + 1],
                                     newDistances[-1])))
    distances = newDistances
  return distances[-1]

try:
  # Cython implementation about 100 time faster
  import Levenshtein
  Levenshtein_distance = Levenshtein.distance
except ImportError as e:
  # python implementation
  Levenshtein_distance = _LevenshteinDistance

edit_distance = Levenshtein_distance

def LER(y_true, y_pred, return_mean=True):
  ''' This function calculates the Labelling Error Rate (PER) of the decoded
  networks output sequence (out) and a target sequence (tar) with Levenshtein
  distance and dynamic programming. This is the same algorithm as commonly used
  for calculating the word error rate (WER), or phonemes error rate (PER).

  Parameters
  ----------
  y_true : ndarray (nb_samples, seq_labels)
      true values of sequences
  y_pred : ndarray (nb_samples, seq_labels)
      prediction values of sequences

  Returns
  -------
  return : float
      Labelling error rate
  '''
  if not hasattr(y_true[0], '__len__') or isinstance(y_true[0], str):
    y_true = [y_true]
  if not hasattr(y_pred[0], '__len__') or isinstance(y_pred[0], str):
    y_pred = [y_pred]

  results = []
  for ytrue, ypred in zip(y_true, y_pred):
    results.append(Levenshtein_distance(ytrue, ypred) / len(ytrue))
  if return_mean:
    return np.mean(results)
  return results

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

  Return an array of interpolated values within given range: [0., 1.]
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
