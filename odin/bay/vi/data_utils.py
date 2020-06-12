from __future__ import absolute_import, division, print_function

import random

import numpy as np
import tensorflow as tf
from six import string_types

from odin.utils import as_tuple


# ===========================================================================
# Helpers
# ===========================================================================
def _fast_samples_indices(known: np.ndarray, factors: np.ndarray):
  outputs = [-1] * len(known)
  for k_idx in range(len(known)):
    for f_idx in range(len(factors)):
      if np.array_equal(known[k_idx], factors[f_idx]):
        if outputs[k_idx] < 0:
          outputs[k_idx] = f_idx
        elif bool(random.getrandbits(1)):
          outputs[k_idx] = f_idx
  return outputs


try:
  # with    numba: ~1.3 sec
  # without numba: ~19.3 sec
  # ~15 times faster
  from numba import jit
  _fast_samples_indices = jit(_fast_samples_indices,
                              target='cpu',
                              cache=False,
                              parallel=False,
                              nopython=True)
except ImportError:
  pass


# ===========================================================================
# Main class
# ===========================================================================
class Factor(object):
  r""" Discrete factor for disentanglement analysis
  If the factors is continuous, the values are casted to `int64`
  For discretizing continuous factor `odin.bay.vi.discretizing`

  Arguments:
    factors : `[num_samples, num_factors]`, an Integer array
    factor_names : None or `[num_factors]`, list of name for each factor
    random_state : an Integer or `np.ranomd.RandomState`

  Attributes:
    factor_labels : list of array, unique labels for each factor
    factor_sizes : list of Integer, number of factor for each factor

  Reference:
    Google research: https://github.com/google-research/disentanglement_lib
  """

  def __init__(self, factors, factor_names=None, random_state=1234):
    if isinstance(factors, tf.data.Dataset):
      factors = tf.stack([x for x in factors])
    if tf.is_tensor(factors):
      factors = factors.numpy()
    factors = np.atleast_2d(factors).astype(np.int64)
    if factors.ndim > 2:
      raise ValueError(
          "factors must be a matrix [n_obeservations, n_factor], but given shape: %s"
          % str(factors.shape))
    num_factors = factors.shape[1]
    # factor_names
    if factor_names is None:
      factor_names = ['F%d' % i for i in range(num_factors)]
    else:
      if hasattr(factor_names, 'numpy'):
        factor_names = factor_names.numpy()
      if hasattr(factor_names, 'tolist'):
        factor_names = factor_names.tolist()
      factor_names = tf.nest.flatten(factor_names)
      assert all(isinstance(i, string_types) for i in factor_names), \
        "All factors' name must be string types, but given: %s" % \
          str(factor_names)
    # store the attributes
    self.factors = factors
    self.factor_names = [str(i) for i in factor_names]
    self.factor_labels = [np.unique(x) for x in factors.T]
    self.factor_sizes = [len(lab) for lab in self.factor_labels]
    if not isinstance(random_state, np.random.RandomState):
      random_state = np.random.RandomState(seed=random_state)
    self.random_state = random_state

  def __str__(self):
    text = f'Factor: {self.factors.shape}\n'
    for name, labels in zip(self.factor_names, self.factor_labels):
      text += " [%d]'%s': %s\n" % (len(labels), name, ', '.join(
          [str(i) for i in labels]))
    return text

  def __repr__(self):
    return self.__str__()

  @property
  def shape(self):
    return self.factors.shape

  @property
  def num_factors(self):
    return len(self.factor_sizes)

  def sample_factors(self,
                     known={},
                     num=16,
                     replace=False,
                     return_indices=False,
                     random_state=None):
    r"""Sample a batch of factors with output shape `[num, num_factor]`.

    Arguments:
      known : A Dictionary, mapping from factor_names/factor_index to
        factor_value/factor_value_index, this establishes a list of known
        factors to sample from the unknown factors.
      num : An Integer
      replace : A Boolean
      return_indices : A Boolean
      random_state : None or `np.random.RandomState`

    Returns:
      factors : `[num, num_factors]`
      indices (optional) : list of Integer
    """
    if random_state is None:
      random_state = self.random_state
    if not isinstance(known, dict):
      known = dict(known)
    known = {
        self.factor_names.index(k)
        if isinstance(k, string_types) else int(k): v \
          for k, v in known.items()
    }
    # make sure value of known factor is the actual label
    for idx, val in list(known.items()):
      labels = self.factor_labels[idx]
      if val not in labels:
        val = labels[val]
      known[idx] = val
    # all samples with similar known factors
    samples = [(idx, x[None, :])
               for idx, x in enumerate(self.factors)
               if all(x[k] == v for k, v in known.items())]
    indices = random_state.choice(len(samples), size=int(num), replace=replace)
    factors = np.vstack([samples[i][1] for i in indices])
    if return_indices:
      return factors, np.array([samples[i][0] for i in indices])
    return factors

  def sample_indices_from_factors(self, factors, random_state=None):
    r"""Sample a batch of observations indices given a batch of factors.
      In other words, the algorithm find all the samples with matching factor
      in given batch, then return the indices of those samples.

    Arguments:
      factors : `[num_samples, num_factors]`
      random_state : None or `np.random.RandomState`

    Returns:
      indices : list of Integer
    """
    if random_state is None:
      random_state = self.random_state
    random.seed(random_state.randint(1e8))
    if factors.ndim == 1:
      factors = np.expand_dims(factors, axis=0)
    assert factors.ndim == 2, "Only support matrix as factors."
    return np.array(_fast_samples_indices(factors, self.factors))
