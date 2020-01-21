from __future__ import absolute_import, division, print_function

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
        newDistances.append(1 + min((distances[index1], distances[index1 + 1],
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
