#
#
from __future__ import absolute_import, division, print_function

import itertools
from numbers import Number
from typing import Union

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
from six import string_types
from typing_extensions import Literal


def _nan_policy(mtx, policy):
  nan_mask = np.isnan(mtx)
  # no NaN values
  if not np.any(nan_mask):
    return mtx
  # given string value
  if isinstance(policy, string_types):
    policy = str(policy).lower().strip()
    if policy == 'propagate':
      return np.nan
    elif policy == 'omit':
      return mtx
    elif policy == 'raise':
      raise RuntimeError(f"Matrix of shape={mtx.shape} contains NaN values.")
  # given number replacement
  elif isinstance(policy, Number):
    return np.where(nan_mask, np.cast[mtx.dtype](policy), mtx)
  raise ValueError(f"Invalid nan_policy= {policy} - {type(policy)}, "
                   "support: 'propagate', 'omit', 'raise' or a number.")


@njit()
def diagonal_bruteforce_search(matrix):
  r""" Find the best permutation of columns to maximize the summarization of
  diagonal entries.

  This algorithm use Heap's algorithm to iterate over all possible permutations
  of columns and store the best `sum(diag(matrix))` and best permutation.

  It guarantee finding the best solution.

  Time complexity is `n!`, recommended for n < 12,
  i.e. about 60s on Intel(R) Xeon(R) CPU E5-1630 v4 @ 3.70GHz

  The function is acclerated by numba which decrease the duration by at least
  5 times.

  Return:
    indices : array
      the columns order that give the maximum diagonal sum

  Reference:
    Heap's Algorithm: https://en.wikipedia.org/wiki/Heap%27s_algorithm
  """
  A = list(range(matrix.shape[1]))
  n = len(A)
  min_dim = min(matrix.shape)
  # prepare the diagonal info
  best_diag = 0
  for i in range(min_dim):
    best_diag += matrix[i, i]
  # this equal to: sum(matrix[i, i] for i in range(min_dim))
  best_perm = list(range(matrix.shape[1]))
  # c is an encoding of the stack state. c[k] encodes the for-loop counter
  # for when permutations(k+1, A) is called
  c = [0 for i in range(n)]
  # i acts similarly to the stack pointer
  i = 0
  while i < n:
    if c[i] < i:
      if i % 2 == 0:
        t = A[0]
        A[0] = A[i]
        A[i] = t
      else:
        t = A[c[i]]
        A[c[i]] = A[i]
        A[i] = t
      # recalculate the best diagonal
      diag = 0
      for row, col in enumerate(A[:min_dim]):
        diag += matrix[row, col]
      # this equal to: diag = sum(matrix[i, j] for i, j in enumerate(A[:min_dim]))
      if diag > best_diag:
        best_diag = diag
        for j, val in enumerate(A):
          best_perm[j] = val
      # Swap has occurred ending the for-loop. Simulate the increment of the
      # for-loop counter
      c[i] += 1
      # Simulate recursive call reaching the base case by bringing the pointer
      # to the base case analog in the array
      i = 0
    else:
      # Calling permutations(i+1, A) has ended as the for-loop terminated.
      # Reset the state and simulate popping the stack by incrementing the
      # pointer.
      c[i] = 0
      i += 1
  return best_perm


def diagonal_linear_assignment(
    matrix: np.ndarray,
    nan_policy: Union[
      Number, Literal['propagate', 'raise', 'omit']] = 'propagate'):
  """ Solve the diagonal linear assignment problem using the
  Hungarian algorithm, this version find the best permutation of columns
  (instead of rows).

  Parameters
  ----------
  matrix : np.ndarray
      a matrix
  nan_policy : {'propagate', 'raise', 'omit'}, optional
      Defines how to handle when input contains nan.
      The following options are available (default is 'propagate'):
        - 'propagate': returns nan
        - 'raise': throws an error
        - 'omit': performs the calculations ignoring nan values
        - `Number`: replace all NaN values with given number

  Return
  ------
  indices : array
    the columns order that give the maximum diagonal sum
  """
  matrix = _nan_policy(matrix, nan_policy)
  # NaN values
  if not hasattr(matrix, 'shape'):
    return matrix
  nrow, ncol = matrix.shape
  if nrow > ncol:
    matrix = matrix[:ncol]
  indices = linear_sum_assignment(matrix.T, maximize=True)
  if nrow < ncol:
    indices = indices[0][np.argsort(indices[1])]
    indices = indices.tolist()
    for i in range(ncol):
      if i not in indices:
        indices.append(i)
    indices = np.array(indices)
  else:
    indices = np.argsort(indices[1])
  return indices


def diagonal_greedy_search(matrix, nonzeros=False):
  r""" Find the best permutation of columns to maximize the summization of
  diagonal entries.

  This algorithm use greedy search looking for the maximum pair `(row, col)`
  for each alignment.

  Return:
    indices : array
      the columns order that give the maximum diagonal sum
  """
  matrix = np.copy(matrix)
  best_perm = [i for i in range(matrix.shape[1])]
  for _ in range(min(matrix.shape)):
    # column with highest max
    max_col = np.argmax(np.max(matrix, axis=0))
    max_row = np.argmax(matrix[:, max_col])
    best_perm[max_row] = max_col
    matrix[:, max_col] = -np.inf
    matrix[max_row, :] = -np.inf
  return best_perm


def diagonal_hillclimb_search(matrix):
  r""" Find the best permutation of columns to maximize the summization of
  diagonal entries.

  This is beam search with `beam_size=1`, this version could perform better
  than greedy search in some case.

  Return:
    indices : array
      the columns order that give the maximum diagonal sum
  """
  return diagonal_beam_search(matrix, beam_size=1)


def diagonal_beam_search(matrix, beam_size=-1):
  r""" Find the best permutation of columns to maximize the summization of
  diagonal entries.

  This is a more strict version of beam search since each beam cannot contain
  duplicated element.

  The implementation is optimized for speed (not memory), the memory complexity
  is: `O(beam_size * matrix.shape[1])`

  Return:
    indices : array
      the columns order that give the maximum diagonal sum
  """
  ncol = matrix.shape[1]
  min_dim = min(matrix.shape)
  if beam_size <= 0:
    beam_size = ncol
  # TODO: in theory beam_size could be larger than dictionary size, but
  # it would complicating the implementation.
  assert beam_size <= ncol, "Beam size must smaller than dictionary"
  beam_seq = np.empty(shape=(beam_size, ncol), dtype=np.int64)
  beam_score = [0. for i in range(beam_size)]
  step_score = np.empty(shape=(beam_size * ncol, ncol + 1), dtype=np.float32)
  # first row
  order = np.argsort(matrix[0])[::-1]
  beam_seq[:, 0] = order[:beam_size]
  beam_score[:] = matrix[0][beam_seq[:, 0]]
  # iterate each column
  for i in range(1, min_dim):
    row = matrix[i]
    order = np.argsort(row)[::-1]
    n = 0
    for beam, score in zip(beam_seq, beam_score):
      for col_idx in order:
        if col_idx not in beam[:i]:
          step_score[n, :ncol] = beam  # store the prev sequence
          step_score[n, i] = col_idx  # next decode value
          step_score[n, -1] = score + row[col_idx]  # total score
          n += 1
    # best solutions
    for j, beam in enumerate(
        sorted(step_score[:n], key=lambda x: x[-1], reverse=True)[:beam_size]):
      beam_seq[j] = beam[:ncol]
      beam_score[j] = beam[-1]
  # add the last dimensions
  if min_dim < ncol:
    for beam in beam_seq:
      idx = min_dim
      for i in range(ncol):
        if i not in beam[:idx]:
          beam[idx] = i
          idx += 1
  return [int(i) for i in beam_seq[0]]
