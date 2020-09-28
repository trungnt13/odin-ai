from __future__ import absolute_import, annotations, division, print_function

import random
import warnings
from collections import Counter
from functools import partial
from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from odin.bay.distributions import CombinedDistribution
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.utils import discretizing
from odin.utils import as_tuple
from six import string_types
from tensorflow.python import keras
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow_probability.python.distributions import (Distribution,
                                                         Independent, Normal,
                                                         VectorDeterministic)
from tqdm import tqdm
from typing_extensions import Literal

__all__ = ['Factor', 'VariationalPosterior']


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


def prepare_inputs_factors(inputs, latents, factors, verbose):
  if inputs is None:
    if latents is None:
      raise ValueError("Either inputs or latents must be provided")
    assert factors is not None, \
      "If latents is provided directly, factors must not be None."
    latents = tf.nest.flatten(latents)
    assert all(isinstance(z, Distribution) for z in latents), \
      ("All latents must be instance of Distribution but given: "
       f"{[type(z).__name__ for z in latents]}")
  ### inputs is a tensorflow Dataset, convert everything to numpy
  elif isinstance(inputs, tf.data.Dataset):
    struct = tf.data.experimental.get_structure(inputs)
    if isinstance(struct, dict):
      struct = struct['inputs']
    struct = tf.nest.flatten(struct)
    n_inputs = len(struct)
    if verbose:
      inputs = tqdm(inputs, desc="Reading data")
    if factors is None:  # include factors
      assert n_inputs >= 2, \
        "factors are not included in the dataset: %s" % str(inputs)
      x, y = [list() for _ in range((n_inputs - 1))], []
      for data in inputs:
        if isinstance(data, dict):  # this is an ad-hoc hack
          data = data['inputs']
        for i, j in enumerate(data[:-1]):
          x[i].append(j)
        y.append(data[-1])
      inputs = [tf.concat(i, axis=0).numpy() for i in x]
      if n_inputs == 2:
        inputs = inputs[0]
      factors = tf.concat(y, axis=0).numpy()
    else:  # factors separated
      x = [list() for _ in range(n_inputs)]
      for data in inputs:
        for i, j in enumerate(tf.nest.flatten(data)):
          x[i].append(j)
      inputs = [tf.concat(i, axis=0).numpy() for i in x]
      if n_inputs == 1:
        inputs = inputs[0]
      if isinstance(factors, tf.data.Dataset):
        if verbose:
          factors = tqdm(factors, desc="Reading factors")
        factors = tf.concat([i for i in factors], axis=0)
    # end the progress
    if isinstance(inputs, tqdm):
      inputs.clear()
      inputs.close()
  # post-processing
  else:
    inputs = tf.nest.flatten(inputs)
  assert len(factors.shape) == 2, "factors must be a matrix"
  return inputs, latents, factors


def _boostrap_sampling(vae: VariationalAutoencoder, inputs: List[np.ndarray],
                       factors: Factor,
                       reduce_latents: Callable[[List[Distribution]],
                                                List[Distribution]],
                       n_samples: int, batch_size: int, verbose: bool,
                       seed: int):
  from odin.bay.helpers import concat_distributions
  inputs = as_tuple(inputs)
  Xs = [list() for _ in range(len(inputs))]  # inputs
  Ys = []  # factors
  Zs = []  # latents
  Os = []  # outputs
  indices = []
  n = 0
  prog = tqdm(desc=f'Sampling', total=n_samples, disable=not verbose)
  while n < n_samples:
    batch = min(batch_size, n_samples - n, factors.shape[0])
    if verbose:
      prog.update(batch)
    # factors
    y, ids = factors.sample_factors(num=batch, return_indices=True, seed=seed)
    indices.append(ids)
    Ys.append(y)
    # inputs
    inps = []
    for xi, inp in zip(Xs, inputs):
      if tf.is_tensor(inp):
        inp = tf.gather(inp, indices=ids, axis=0)
      else:
        inp = inp[ids]
      xi.append(inp)
      inps.append(inp)
    # latents representation
    z = vae.encode(inps, training=False)
    o = tf.nest.flatten(as_tuple(vae.decode(z, training=False)))
    # post-process latents
    z = reduce_latents(as_tuple(z))
    if len(z) == 1:
      z = z[0]
    Os.append(o)
    Zs.append(z)
    # update the counter
    n += len(y)
  # end progress
  prog.clear()
  prog.close()
  # aggregate all data
  Xs = [np.concatenate(x, axis=0) for x in Xs]
  Ys = np.concatenate(Ys, axis=0)
  if isinstance(Zs[0], Distribution):
    Zs = concat_distributions(Zs, name="Latents")
  else:
    Zs = CombinedDistribution([
        concat_distributions([z[zi]
                              for z in Zs], name=f"Latents{zi}")
        for zi in range(len(Zs[0]))
    ],
                              name="Latents")
  Os = [
      concat_distributions([j[i]
                            for j in Os], name=f"Output{i}")
      for i in range(len(Os[0]))
  ]
  return Xs, Ys, Zs, Os, np.concatenate(indices, axis=0)


# ===========================================================================
# Factor
# ===========================================================================
class Factor:
  """Discrete factor for disentanglement analysis. If the factors is continuous,
  the values are casted to `int64` For discretizing continuous factor
  `odin.bay.vi.discretizing`

  Parameters
  ----------
  factors : [type]
      `[num_samples, num_factors]`, an Integer array
  factor_names : [type], optional
      None or `[num_factors]`, list of name for each factor, by default None

  Attributes
  ---------
      factor_labels : list of array, unique labels for each factor
      factor_sizes : list of Integer, number of factor for each factor

  Reference
  ---------
      Google research: https://github.com/google-research/disentanglement_lib

  Raises
  ------
  ValueError
      factors must be a matrix
  """

  def __init__(self,
               factors: np.ndarray,
               factor_names: Optional[List[str]] = None):
    if isinstance(factors, tf.data.Dataset):
      factors = tf.stack([x for x in factors])
    if tf.is_tensor(factors):
      factors = factors.numpy()
    factors = np.atleast_2d(factors).astype(np.int64)
    if factors.ndim > 2:
      raise ValueError("factors must be a matrix [n_observations, n_factor], "
                       f"but given shape:{factors.shape}")
    num_factors = factors.shape[1]
    # factor_names
    if factor_names is None:
      factor_names = [f'F{i}' for i in range(num_factors)]
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

  def __str__(self):
    text = f'Factor: {self.factors.shape}\n'
    for name, labels in zip(self.factor_names, self.factor_labels):
      text += " [%d]'%s': %s\n" % (len(labels), name, ', '.join(
          [str(i) for i in labels]))
    return text[:-1]

  def __repr__(self):
    return self.__str__()

  @property
  def shape(self) -> List[int]:
    return self.factors.shape

  @property
  def num_factors(self) -> int:
    return len(self.factor_sizes)

  def sample_factors(self,
                     known: Dict[str, int] = {},
                     num: int = 16,
                     replace: bool = False,
                     return_indices: bool = False,
                     seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Sample a batch of factors with output shape `[num, num_factor]`.

    Arguments:
      known : A Dictionary, mapping from factor_names|factor_index to
        factor_value|factor_value_index, this establishes a list of known
        factors to sample from the unknown factors.
      num : An Integer
      replace : A Boolean
      return_indices : A Boolean

    Returns:
      factors : `[num, num_factors]`
      indices (optional) : list of Integer
    """
    random_state = np.random.RandomState(seed=seed)
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

  def sample_indices_from_factors(self,
                                  factors: np.ndarray,
                                  seed: int = 1) -> np.ndarray:
    r"""Sample a batch of observations indices given a batch of factors.
      In other words, the algorithm find all the samples with matching factor
      in given batch, then return the indices of those samples.

    Arguments:
      factors : `[num_samples, num_factors]`
      random_state : None or `np.random.RandomState`

    Returns:
      indices : list of Integer
    """
    random_state = np.random.RandomState(seed=1)
    random.seed(random_state.randint(1e8))
    if factors.ndim == 1:
      factors = np.expand_dims(factors, axis=0)
    assert factors.ndim == 2, "Only support matrix as factors."
    return np.array(_fast_samples_indices(factors, self.factors))


# ===========================================================================
# Sampler
# ===========================================================================
class Posterior:

  @property
  def model(self) -> keras.layers.Layer:
    raise NotImplementedError

  def copy(self) -> Posterior:
    raise NotImplementedError


class VariationalPosterior(Posterior):
  """Posterior class for variational inference using Variational Autoencoder"""

  def __init__(self,
               vae: VariationalAutoencoder,
               inputs: Optional[Union[np.ndarray, tf.Tensor, DatasetV2]] = None,
               latents: Optional[Union[np.ndarray, tf.Tensor,
                                       DatasetV2]] = None,
               factors: Optional[Union[np.ndarray, tf.Tensor,
                                       DatasetV2]] = None,
               discretizer: Optional[Callable[[np.ndarray],
                                              np.ndarray]] = partial(
                                                  discretizing,
                                                  n_bins=5,
                                                  strategy='quantile'),
               factor_names: Optional[List[str]] = None,
               n_samples: int = 5000,
               batch_size: int = 32,
               reduce_latents: Callable[[List[Distribution]], List[Distribution]] = \
                 lambda x: x,
               verbose: bool = False,
               seed: int = 1):
    super().__init__()
    assert isinstance(vae, VariationalAutoencoder), \
      ("vae must be instance of odin.bay.vi.VariationalAutoencoder, "
       f"given: {type(vae)}")
    assert callable(reduce_latents), 'reduce_latents function must be callable'
    ### Assign basic attributes
    self._vae = vae
    self.reduce_latents = reduce_latents
    self.verbose = bool(verbose)
    #### prepare the sampling
    inputs, latents, factors = prepare_inputs_factors(inputs,
                                                      latents,
                                                      factors,
                                                      verbose=verbose)
    n_inputs = factors.shape[0]
    n_factors = factors.shape[1]
    if factor_names is None:
      factor_names = np.asarray([f'F{i}' for i in range(n_factors)])
    else:
      assert len(factor_names) == n_factors, \
        f"There are {n_factors} factors, but only given {len(factor_names)} names"
    ## discretized factors
    factors_original = factors
    if discretizer is not None:
      if verbose:
        print("Discretizing factors ...")
      factors = discretizer(factors)
    # check for singular factor and ignore it
    ids = []
    for i, (name, f) in enumerate(zip(factor_names, factors.T)):
      c = Counter(f)
      if len(c) < 2:
        warnings.warn(f"Ignore factor with name '{name}', singular data: {f}")
      else:
        ids.append(i)
    if len(ids) != len(factor_names):
      factors_original = factors_original[:, ids]
      factor_names = factor_names[ids]
      factors = factors[:, ids]
    # create the factor class for sampling
    factors_set = Factor(factors, factor_names=factor_names)
    ## latents are given directly
    if inputs is None:
      latents = self.reduce_latents(as_tuple(latents))
      if len(latents) == 1:
        latents = latents[0]
      else:
        latents = CombinedDistribution(latents, name="Latents")
      outputs = None
      indices = None
    ## sampling the latents
    else:
      inputs, factors, latents, outputs, indices = \
          _boostrap_sampling(self.model,
                         inputs=inputs,
                         factors=factors_set,
                         batch_size=batch_size,
                         n_samples=n_samples,
                         reduce_latents=reduce_latents,
                         verbose=verbose,
                         seed=seed)
    ## assign the attributes
    self._factors_original = factors_original.numpy() if tf.is_tensor(
        factors_original) else factors_original
    self._inputs = inputs
    self._factors = factors
    self._latents = latents
    self._outputs = outputs
    self._indices = indices
    self._factor_names = factors_set.factor_names

  @property
  def model(self) -> VariationalAutoencoder:
    return self._vae

  @property
  def inputs(self) -> List[np.ndarray]:
    return self._inputs

  @property
  def latents(self) -> Distribution:
    r""" Return the learned latent representations `Distribution`
    (i.e. the latent code) for training and testing """
    return self._latents

  @property
  def outputs(self) -> List[Distribution]:
    r""" Return the reconstructed `Distributions` of inputs for training and
    testing """
    return self._outputs

  @property
  def factors(self) -> np.ndarray:
    r""" Return the target variable (i.e. the factors of variation) for
    training and testing """
    return self._factors

  @property
  def factors_original(self) -> np.ndarray:
    r"""Return the original factors, i.e. the factors before discretizing """
    # the original factors is the same for all samples set
    return self._factors_original

  @property
  def n_factors(self) -> int:
    return self.factors[0].shape[1]

  @property
  def n_latents(self) -> int:
    r""" return the number of latent codes """
    return self.latents.event_shape[0]

  @property
  def n_samples(self) -> int:
    r""" Return number of samples for testing """
    return self.latents.batch_shape[0]

  @property
  def factor_names(self) -> List[str]:
    return self._factor_names

  @property
  def code_names(self) -> List[str]:
    return [f"Z{i}" for i in range(self.n_latents)]

  ############## Experiment setup
  def traverse(self,
               min_val: int = -2.0,
               max_val: int = 2.0,
               num: int = 11,
               n_samples: int = 1,
               mode: Literal['linear', 'quantile', 'gaussian'] = 'linear',
               convert_to_tensor: Callable[[Distribution],
                                           tf.Tensor] = lambda d: d.mean(),
               seed: int = 1) -> VariationalPosterior:
    """Create data for latents' traverse experiments

    Parameters
    ----------
    min_val : int, optional
        [description], by default -2.0
    max_val : int, optional
        [description], by default 2.0
    num : int, optional
        [description], by default 11
    n_samples : int, optional
        [description], by default 2
    mode : {'linear', 'quantile', 'gaussian'}, optional
        [description], by default 'linear'
    convert_to_tensor : Callable[[Distribution], tf.Tensor], optional
        [description], by default lambdad:d.mean()

    Returns
    -------
    VariationalPosterior
        a copy of VariationalPosterior with the new traversed latents,
        the number of sample is: `n_samples * num * n_latents`

    Example
    --------
    For `n_samples=2`, `num=2`, and `n_latents=2`, the return latents are:
    ```
    [[-2., 0.47],
     [ 0., 0.47],
     [ 2., 0.47],
     [-2., 0.31],
     [ 0., 0.31],
     [ 2., 0.31],
     [0.14, -2.],
     [0.14,  0.],
     [0.14,  2.],
     [0.91, -2.],
     [0.91,  0.],
     [0.91,  2.]]
    ```
    """
    num = int(num)
    assert num % 2 == 1, f'num must be odd number, i.e. centerred at 0, given {num}'
    n_samples = int(n_samples)
    assert num > 1 and n_samples > 0, \
      ("num > 1 and n_samples > 0, "
       f"but given: num={num} n_samples={n_samples}")
    # ====== check the mode ====== #
    all_mode = ('quantile', 'linear', 'gaussian')
    mode = str(mode).strip().lower()
    assert mode in all_mode, \
      f"Only support mode:{all_mode}, but given mode='{mode}'"
    ### sample
    random_state = np.random.RandomState(seed=seed)
    indices = random_state.choice(self.n_samples, size=n_samples, replace=False)
    Z_org = convert_to_tensor(self.latents).numpy()
    Z = Z_org[indices]
    ### ranges
    # z_range is a matrix [n_latents, num]
    # linear range
    if mode == 'linear':
      x = np.expand_dims(np.linspace(min_val, max_val, num), axis=0)
      z_range = np.repeat(x, self.n_latents, axis=0)
    # min-max quantile
    elif mode == 'quantile':
      z_range = []
      for vmin, vmax in zip(np.min(Z_org, axis=0), np.max(Z_org, axis=0)):
        z_range.append(np.expand_dims(np.linspace(vmin, vmax, num=num), axis=0))
      z_range = np.concatenate(z_range, axis=0)
    # gaussian quantile
    elif mode == 'gaussian':
      dist = Normal(loc=tf.reduce_mean(self.latents.mean(), 0),
                    scale=tf.reduce_mean(self.latents.stddev(), 0))
      z_range = []
      for i in np.linspace(1e-5, 1.0 - 1e-5, num=num, dtype=np.float32):
        z_range.append(np.expand_dims(dist.quantile(i), axis=1))
      z_range = np.concatenate(z_range, axis=1)
    ### traverse
    Zs = []
    Z_indices = []
    for i, zr in enumerate(z_range):
      z_i = np.repeat(np.array(Z), len(zr), axis=0)
      Z_indices.append(np.repeat(indices, len(zr), axis=0))
      # repeat for each sample
      for j in range(n_samples):
        s = j * len(zr)
        e = (j + 1) * len(zr)
        z_i[s:e, i] = zr
      Zs.append(z_i)
    Zs = np.concatenate(Zs, axis=0)
    Z_indices = np.concatenate(Z_indices, axis=0)
    ### create the new posterior
    # NOTE: this might not work for multi-latents
    outputs = list(as_tuple(self.model.decode(Zs, training=False)))
    obj = self.copy(Z_indices,
                    latents=VectorDeterministic(Zs, name="Latents"),
                    outputs=outputs)
    return obj

  def conditioning(self,
                   known: Dict[Union[str, int], Callable[[int], bool]],
                   logical_not: bool = False,
                   n_samples: Optional[int] = None,
                   seed: int = 1) -> VariationalPosterior:
    r""" Conditioning the sampled dataset on known factors

    Arguments:
      known : a mapping from index or name of factor to a callable, the
        callable must return a list of boolean indices, which indicates
        the samples to be selected
      logical_not : a Boolean, if True applying the opposed conditioning
        of the known factors
      n_samples : an Integer (Optional), maximum number of selected samples.

    Return:
      a new `Criticizer` with the conditioned data and representations

    Example:
    ```
    # conditioning on: (1st-factor > 2) and (2nd-factor == 3)
    conditioning({1: lambda x: x > 2, 2: lambda x: x==3})
    ```
    """
    known = {
        int(k) if isinstance(k, Number) else self.factor_names.index(str(k)): v
        for k, v in dict(known).items()
    }
    assert len(known) > 0 and all(callable(v) for v in known.values()), \
      ("known factors must be mapping from factor index to callable "
        f"but given: {known}")
    # start conditioning
    ids = np.full(shape=self.n_samples, fill_value=True, dtype=np.bool)
    for f_idx, fn_filter in known.items():
      ids = np.logical_and(ids, fn_filter(self.factors[:, f_idx]))
    # opposing the conditions
    if logical_not:
      ids = np.logical_not(ids)
    ids = np.arange(self.n_samples, dtype=np.int32)[ids]
    # select n_samples
    if n_samples is not None:
      random_state = np.random.RandomState(seed=seed)
      n_samples = int(n_samples)
      ids = random_state.choice(ids,
                                size=n_samples,
                                replace=n_samples > len(ids))
    # copy the posterior
    obj = VariationalPosterior.__new__(VariationalPosterior)
    obj._vae = self._vae
    obj._factor_names = list(self.factor_names)
    obj.reduce_latents = self.reduce_latents
    obj.verbose = self.verbose
    # slice the data
    obj._factors_original = self.factors_original[ids]
    obj._factors = self.factors[ids]
    obj._inputs = [x[ids] for x in self.inputs]
    obj._indices = self._indices[ids]
    # convert boolean indices to integer
    z = as_tuple(self.model.encode(obj.inputs, training=False))
    z = self.reduce_latents(z)
    if len(z) > 1:
      z = CombinedDistribution(z, name='Latents')
    else:
      z = z[0]
    obj._latents = z
    obj._outputs = list(as_tuple(self.model.decode(z, training=False)))
    return obj

  def copy(
      self,
      indices: Optional[Union[slice, List[int]]] = None,
      latents: Optional[Distribution] = None,
      outputs: Optional[List[Distribution]] = None) -> VariationalPosterior:
    """Return the deepcopy"""
    obj = VariationalPosterior.__new__(VariationalPosterior)
    obj._vae = self._vae
    obj._factor_names = list(self.factor_names)
    obj.reduce_latents = self.reduce_latents
    obj.verbose = self.verbose
    # helper for slicing
    fslice = lambda x: x[indices] if indices is not None else x
    # copy the factors
    print(self._factors_original)
    obj._factors_original = np.array(fslice(self._factors_original))
    obj._factors = np.array(fslice(self._factors))
    # copy the inputs
    obj._inputs = [np.array(fslice(i)) for i in self._inputs]
    obj._indices = np.array(fslice(self._indices))
    # copy the latents and outputs
    if indices is not None:
      assert latents is not None and isinstance(latents, Distribution), \
        f"Invalid latents type {type(latents)}"
      obj._latents = latents
      if outputs is None:
        obj._outputs = list(as_tuple(self.model.decode(latents,
                                                       training=False)))
      else:
        obj._outputs = list(as_tuple(outputs))
    else:
      obj._latents = self._latents.copy()
      obj._outputs = [o.copy() for o in self._outputs]
    return obj

  def __str__(self):
    dname = lambda d: str(d).replace('Independent',
                                     d.distribution.__class__.__name__) \
      if isinstance(d, Independent) else str(d)
    return \
f"""Variational Posterior:
  model  : {self._vae.__class__}
  reduce : {self.reduce_latents}
  verbose: {self.verbose}
  factors: {self.factors.shape} - {', '.join(self.factor_names)}
  inputs : {', '.join(str((i.shape, i.dtype)) for i in self.inputs)}
  outputs: {', '.join(dname(o).replace('tfp.distributions.', '') for o in self.outputs)}
  latents: {dname(self.latents).replace('tfp.distributions.', '')}"""
