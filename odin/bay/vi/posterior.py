from __future__ import absolute_import, annotations, division, print_function

import random
import warnings
from collections import Counter, OrderedDict
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
import tensorflow as tf
from numpy import ndarray
from odin import visual as vs
from odin.bay.distributions import CombinedDistribution
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.metrics import (mutual_info_estimate, mutual_info_gap,
                                 representative_importance_matrix)
from odin.bay.vi.utils import discretizing
from odin.ml import dimension_reduce, linear_classifier
from odin.search import diagonal_linear_assignment
from odin.utils import as_tuple
from six import string_types
from tensorflow import Tensor
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
def _fast_samples_indices(known: ndarray, factors: ndarray):
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
  _fast_samples_indices = jit(
      _fast_samples_indices,
      # target='cpu',
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
      assert n_inputs >= 2, f"factors are not included in the dataset: {inputs}"
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


def _boostrap_sampling(vae: VariationalAutoencoder, inputs: List[ndarray],
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
               factors: ndarray,
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
                     seed: int = 1) -> Tuple[ndarray, ndarray]:
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
                                  factors: ndarray,
                                  seed: int = 1) -> ndarray:
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
class Posterior(vs.Visualizer):

  def plot_scatter(
      self,
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      classifier: Optional[Literal['svm', 'tree', 'logistic', 'knn', 'lda',
                                   'gbt']] = None,
      classifier_kw: Dict[str, Any] = {},
      dimension_reduction: Literal['pca', 'umap', 'tsne', 'knn',
                                   'kmean'] = 'tsne',
      factor_indices: Optional[Union[int, str, List[Union[int, str]]]] = None,
      n_samples: Optional[int] = None,
      return_figure: bool = False,
      seed: int = 1,
  ):
    cmap = 'bwr'
    ## get all relevant factors
    if factor_indices is None:
      factor_indices = list(range(self.n_factors))
    factor_indices = [
        int(i) if isinstance(i, Number) else self.factor_names.index(i)
        for i in as_tuple(factor_indices)
    ]
    f = self.factors[:, factor_indices]
    if f.ndim == 1:
      f = np.expand_dims(f, axis=1)
    names = np.asarray(self.factor_names)[factor_indices]
    ## reduce latents dimension
    z = convert_to_tensor(self.latents).numpy()
    z = dimension_reduce(z,
                         algo=dimension_reduction,
                         n_components=2,
                         random_state=seed)
    x_min, x_max = np.min(z[:, 0]), np.max(z[:, 0])
    y_min, y_max = np.min(z[:, 1]), np.max(z[:, 1])
    # standardlize the factors
    f_norm = (f - np.mean(f, axis=0, keepdims=True)) / np.std(
        f, axis=0, keepdims=True)
    ## downsample
    if isinstance(n_samples, Number):
      n_samples = int(n_samples)
      if n_samples < z.shape[0]:
        rand = np.random.RandomState(seed=seed)
        ids = rand.choice(np.arange(z.shape[0], dtype=np.int32),
                          size=n_samples,
                          replace=False)
        z = z[ids]
        f = f[ids]
    n_samples = z.shape[0]
    if classifier is not None:
      xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_samples),
                           np.linspace(y_min, y_max, n_samples))
      xy = np.c_[xx.ravel(), yy.ravel()]
    ## plotting
    from matplotlib import pyplot as plt
    n_cols = 4
    n_rows = int(np.ceil(f.shape[1] / n_cols))
    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2),
                     constrained_layout=False,
                     dpi=120)
    grids = fig.add_gridspec(n_rows, n_cols, wspace=0, hspace=0)
    for c in range(n_cols):
      for r in range(n_rows):
        idx = r * n_cols + c
        if idx >= f.shape[1]:
          continue
        ax = fig.add_subplot(grids[r, c])
        # scatter plot
        vs.plot_scatter(x=z,
                        val=f_norm[:, idx],
                        color=cmap,
                        ax=ax,
                        size=10.,
                        alpha=0.5)
        ax.grid(False)
        ax.tick_params(axis='both',
                       bottom=False,
                       top=False,
                       left=False,
                       right=False)
        ax.text(x_min,
                y_max,
                names[idx],
                horizontalalignment='left',
                verticalalignment='top',
                fontdict=dict(size=10,
                              color='Green',
                              alpha=0.5,
                              weight='normal'))
        # classifier boundary
        if classifier is not None:
          model = linear_classifier(z,
                                    f[:, idx],
                                    algo=classifier,
                                    seed=seed,
                                    **classifier_kw)
          ax.contourf(xx,
                      yy,
                      model.predict(xy).reshape(xx.shape),
                      cmap=cmap,
                      alpha=0.4)
    ## return or save
    if return_figure:
      return fig
    return self.add_figure(
        name=f'scatter_{dimension_reduction}_{str(classifier).lower()}',
        fig=fig)

  def plot_histogram(
      self,
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      histogram_bins: int = 120,
      original_factors: bool = True,
      return_figure: bool = False,
  ):
    Z = convert_to_tensor(self.latents).numpy()
    F = self.factors_original if original_factors else self.factors
    X = [i for i in F.T] + [i for i in Z.T]
    labels = self.factor_names + self.latent_names
    # create the figure
    ncol = int(np.ceil(np.sqrt(len(X)))) + 1
    nrow = int(np.ceil(len(X) / ncol))
    fig = vs.plot_figure(nrow=12, ncol=20, dpi=100)
    for i, (x, lab) in enumerate(zip(X, labels)):
      vs.plot_histogram(x,
                        ax=(nrow, ncol, i + 1),
                        bins=int(histogram_bins),
                        title=lab,
                        alpha=0.8,
                        color='blue',
                        fontsize=16)
    fig.tight_layout()
    if return_figure:
      return fig
    return self.add_figure(
        f"histogram_{'original' if original_factors else 'discretized'}", fig)

  def plot_disentanglement(
      self,
      factor_indices: Optional[Union[int, str, List[Union[int, str]]]] = None,
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      n_bins_factors: int = 15,
      n_bins_codes: int = 80,
      corr_type: Union[Literal['spearman', 'pearson', 'lasso', 'average', 'mi'],
                       ndarray] = 'average',
      original_factors: bool = True,
      show_all_codes: bool = False,
      sort_pairs: bool = True,
      title: str = '',
      return_figure: bool = False,
      seed: int = 1,
  ):
    r""" To illustrate the disentanglement of the codes, the codes' histogram
    bars are colored by the value of factors.

    Arguments:
      factor_names : list of String or Integer.
        Name or index of which factors will be used for visualization.
      factor_bins : factor is discretized into bins, then a LogisticRegression
        model will predict the bin (with color) given the code as input.
      corr_type : {'spearman', 'pearson', 'lasso', 'average', 'mi', None, matrix}
        Type of correlation, with special case 'mi' for mutual information.
          - If None, no sorting by correlation provided.
          - If an array, the array must have shape `[n_codes, n_factors]`
      show_all_codes : a Boolean.
        if False, only show most correlated codes-factors, otherwise,
        all codes are shown for each factor.
        This option only in effect when `corr_type` is not `None`.
      original_factors : optional original factors before discretized by
        `Criticizer`
    """
    ### prepare styled plot
    styles = dict(fontsize=12,
                  cbar_horizontal=False,
                  bins_color=int(n_bins_factors),
                  bins=int(n_bins_codes),
                  color='bwr',
                  alpha=0.8)
    # get all relevant factors
    if factor_indices is None:
      factor_indices = list(range(self.n_factors))
    factor_indices = [
        int(i) if isinstance(i, Number) else self.factor_names.index(i)
        for i in as_tuple(factor_indices)
    ]
    ### correlation
    if isinstance(corr_type, string_types):
      if corr_type == 'mi':
        corr = self.mutualinfo_matrix(convert_to_tensor=convert_to_tensor,
                                      seed=seed)
        score_type = 'mutual-info'
      else:
        corr = self.correlation_matrix(convert_to_tensor=convert_to_tensor,
                                       method=corr_type,
                                       seed=seed)
        score_type = corr_type
      # [n_factors, n_codes]
      corr = corr.T[factor_indices]
    ### directly give the correlation matrix
    elif isinstance(corr_type, ndarray):
      corr = corr_type
      if self.n_latents != self.n_factors and corr.shape[0] == self.n_latents:
        corr = corr.T
      assert corr.shape == (self.n_factors, self.n_latents), \
        (f"Correlation matrix expect shape (n_factors={self.n_factors}, "
         f"n_codes={self.n_codes}) but given shape: {corr.shape}")
      score_type = 'score'
      corr = corr[factor_indices]
    ### exception
    else:
      raise ValueError(
          f"corr_type could be string, None or a matrix but given: {type(corr_type)}"
      )
    ### sorting the latents
    if sort_pairs:
      latent_indices = diagonal_linear_assignment(np.abs(corr), nan_policy=0)
    else:
      latent_indices = np.arange(self.n_latents, dtype=np.int32)
    if not show_all_codes:
      latent_indices = latent_indices[:len(factor_indices)]
    corr = corr[:, latent_indices]
    ### prepare the data
    # factors
    F = (self.factors_original
         if original_factors else self.factors)[:, factor_indices]
    factor_names = np.asarray(self.factor_names)[factor_indices]
    # codes
    Z = convert_to_tensor(self.latents).numpy()[:, latent_indices]
    latent_names = np.asarray(self.latent_names)[latent_indices]
    ### create the figure
    nrow = F.shape[1]
    ncol = Z.shape[1] + 1
    fig = vs.plot_figure(nrow=nrow * 3, ncol=ncol * 2.8, dpi=100)
    count = 1
    for fidx, (f, fname) in enumerate(zip(F.T, factor_names)):
      # the first plot show how the factor clustered
      ax, _, _ = vs.plot_histogram(x=f,
                                   color_val=f,
                                   ax=(nrow, ncol, count),
                                   cbar=False,
                                   title=f"{fname}",
                                   **styles)
      ax.tick_params(axis='y', labelleft=False)
      count += 1
      # the rest of the row show how the codes align with the factor
      for zidx, (score, z,
                 zname) in enumerate(zip(corr[fidx], Z.T, latent_names)):
        text = "*" if fidx == zidx else ""
        ax, _, _ = vs.plot_histogram(
            x=z,
            color_val=f,
            ax=(nrow, ncol, count),
            cbar=False,
            title=f"{text}{fname}-{zname} (${score:.2f}$)",
            bold_title=True if fidx == zidx else False,
            **styles)
        ax.tick_params(axis='y', labelleft=False)
        count += 1
    ### fine tune the plot
    fig.suptitle(f"[{score_type}]{title}", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    if return_figure:
      return fig
    return self.add_figure(
        f"disentanglement_{'original' if original_factors else 'discretized'}",
        fig)

  @property
  def model(self) -> keras.layers.Layer:
    raise NotImplementedError

  @property
  def inputs(self) -> List[ndarray]:
    raise NotImplementedError

  @property
  def latents(self) -> Distribution:
    raise NotImplementedError

  @property
  def outputs(self) -> List[Distribution]:
    raise NotImplementedError

  @property
  def factors(self) -> ndarray:
    raise NotImplementedError

  @property
  def factors_original(self) -> ndarray:
    raise NotImplementedError

  @property
  def factor_names(self) -> List[str]:
    raise NotImplementedError

  @property
  def latent_names(self) -> List[str]:
    raise NotImplementedError

  @property
  def n_factors(self) -> int:
    return self.factors.shape[1]

  @property
  def n_latents(self) -> int:
    return self.latents.event_shape[0]

  @property
  def n_samples(self) -> int:
    return self.latents.batch_shape[0]

  ############## Matrices
  def correlation_matrix(
      self,
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      method: Literal['spearman', 'pearson', 'lasso', 'average'] = 'spearman',
      sort_pairs: bool = False,
      seed: int = 1,
  ) -> ndarray:
    """Correlation matrix of `latent codes` (row) and `groundtruth factors`
    (column).

    Parameters
    ----------
    convert_to_tensor : Callable[[Distribution], Tensor], optional
        callable to convert a distribution to tensor, by default `lambdad:d.mean()`
    method : {'spearman', 'pearson', 'lasso', 'average'}
        method for calculating the correlation,
        'spearman' - rank or monotonic correlation
        'pearson' - linear correlation
        'lasso' - lasso regression
        'average' - compute all known method then taking average,
        by default 'spearman'
    sort_pairs : bool, optional
        If True, reorganize the row of correlation matrix
        for the best match between code-factor (i.e. the largest diagonal sum).
        Note: the decoding is performed on train matrix, then applied to test
        matrix, by default False
    seed : int, optional
        random state seed, by default 1

    Returns
    -------
    ndarray
        correlation matrices `[n_latents, n_factors]`, all entries are in `[0, 1]`.
    OrderedDict (optional)
        mapping from decoded factor index to latent code index.
    """
    method = str(method).strip().lower()
    all_corr = ['spearman', 'lasso', 'pearson', 'average']
    assert method in all_corr, \
      f"Support {all_corr} correlation but given method='{method}'"
    ### average mode
    if method == 'average':
      corr_mat = sum(
          self.correlation_matrix(convert_to_tensor=convert_to_tensor,
                                  method=corr,
                                  sort_pairs=False,
                                  seed=seed)
          for corr in ['spearman', 'pearson', 'lasso']) / 3
    ### specific mode
    else:
      # start form correlation matrix
      z = convert_to_tensor(self.latents).numpy()
      f = self.factors
      # lasso
      if method == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(random_state=seed, alpha=0.1)
        model.fit(z, f)
        # coef_ is [n_target, n_features], so we need transpose here
        corr_mat = np.transpose(np.absolute(model.coef_))
      # spearman and pearson
      else:
        corr_mat = np.empty(shape=(self.n_latents, self.n_factors),
                            dtype=np.float64)
        for code in range(self.n_latents):
          for fact in range(self.n_factors):
            x, y = z[:, code], f[:, fact]
            if method == 'spearman':
              corr = sp.stats.spearmanr(x, y, nan_policy="omit")[0]
            elif method == 'pearson':
              corr = sp.stats.pearsonr(x, y)[0]
            corr_mat[code, fact] = corr
    ## decoding and return
    if sort_pairs:
      ids = diagonal_linear_assignment(corr_mat)
      corr_mat = corr_mat[ids, :]
      return corr_mat, OrderedDict(zip(range(self.n_factors), ids))
    return corr_mat

  def mutualinfo_matrix(
      self,
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      n_neighbors: Union[int, List[int]] = [3, 4, 5],
      n_cpu: int = 1,
      seed: int = 1,
  ) -> np.ndarray:
    """Mutual Information estimated between each latents' dimension and factors'
    dimension

    Parameters
    ----------
    convert_to_tensor : Callable[[Distribution], Tensor], optional
        callable to convert a distribution to tensor, by default `lambdad:d.mean()`
    n_neighbors : Union[int, List[int]], optional
        number of neighbors for estimating MI, by default [3, 4, 5]
    n_cpu : int, optional
        number of CPU for parallel, by default 1
    seed : int, optional
        random state seed, by default 1

    Returns
    -------
    np.ndarray
        matrix `[n_latents, n_factors]`, estimated mutual information between
          each representation and each factors
    """
    n_neighbors = as_tuple(n_neighbors, t=int)
    mi = sum(
        mutual_info_estimate(
            representations=convert_to_tensor(self.latents).numpy(),
            factors=self.factors,
            continuous_representations=True,
            continuous_factors=False,
            n_neighbors=i,
            n_cpu=n_cpu,
            seed=seed,
        ) for i in n_neighbors)
    return mi / len(n_neighbors)

  def mutual_info_gap(
      self,
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      n_bins: int = 10,
      strategy: Literal['uniform', 'quantile', 'kmeans', 'gmm'] = 'uniform',
  ) -> float:
    """Mutual Information Gap

    Parameters
    ----------
    convert_to_tensor : Callable[[Distribution], Tensor], optional
        callable to convert a distribution to tensor, by default `lambdad:d.mean()`
    n_bins : int, optional
        number of bins for discretizing the latents, by default 10
    strategy : {'uniform', 'quantile', 'kmeans', 'gmm'}
        Strategy used to define the widths of the bins.
        'uniform' - All bins in each feature have identical widths.
        'quantile' - All bins in each feature have the same number of points.
        'kmeans' - Values in each bin have the same nearest center of a 1D cluster.
        , by default 'uniform'

    Returns
    -------
    float
        mutual information gap score
    """
    z = convert_to_tensor(self.latents).numpy()
    if n_bins > 1:
      z = discretizing(z, independent=True, n_bins=n_bins, strategy=strategy)
    f = self.factors
    return mutual_info_gap(z, f)

  def copy(self, *args, **kwargs) -> Posterior:
    raise NotImplementedError


class VariationalPosterior(Posterior):
  """Posterior class for variational inference using Variational Autoencoder"""

  def __init__(self,
               vae: VariationalAutoencoder,
               inputs: Optional[Union[ndarray, Tensor, DatasetV2]] = None,
               latents: Optional[Union[ndarray, Tensor,
                                       DatasetV2]] = None,
               factors: Optional[Union[ndarray, Tensor,
                                       DatasetV2]] = None,
               discretizer: Optional[Callable[[ndarray],
                                              ndarray]] = partial(
                                                  discretizing,
                                                  n_bins=5,
                                                  strategy='quantile'),
               factor_names: Optional[List[str]] = None,
               n_samples: int = 5000,
               batch_size: int = 32,
               reduce_latents: Callable[[List[Distribution]], List[Distribution]] = \
                 lambda x: x,
               verbose: bool = False,
               seed: int = 1,):
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
  def inputs(self) -> List[ndarray]:
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
  def factors(self) -> ndarray:
    r""" Return the target variable (i.e. the factors of variation) for
    training and testing """
    return self._factors

  @property
  def factors_original(self) -> ndarray:
    r"""Return the original factors, i.e. the factors before discretizing """
    # the original factors is the same for all samples set
    return self._factors_original

  @property
  def factor_names(self) -> List[str]:
    return self._factor_names

  @property
  def latent_names(self) -> List[str]:
    return [f"Z{i}" for i in range(self.n_latents)]

  ############## Experiment setup
  def traverse(
      self,
      min_val: int = -2.0,
      max_val: int = 2.0,
      num: int = 11,
      n_samples: int = 1,
      mode: Literal['linear', 'quantile', 'gaussian'] = 'linear',
      convert_to_tensor: Callable[[Distribution], Tensor] = lambda d: d.mean(),
      seed: int = 1,
  ) -> VariationalPosterior:
    """Create data for latents' traverse experiments

    Parameters
    ----------
    min_val : int, optional
        minimum value of the traverse, by default -2.0
    max_val : int, optional
        maximum value of the traverse, by default 2.0
    num : int, optional
        number of points in the traverse, must be odd number, by default 11
    n_samples : int, optional
        number of samples selected for the traverse, by default 2
    mode : {'linear', 'quantile', 'gaussian'}, optional
        'linear' mode take linear interpolation between the `min_val` and `max_val`.
        'quantile' mode return `num` quantiles based on min and max values inferred
        from the data. 'gaussian' mode takes `num` Gaussian quantiles,
        by default 'linear'
    convert_to_tensor : Callable[[Distribution], Tensor], optional
        function to convert Distribution to tensor, by default `lambda:d.mean()`

    Returns
    -------
    VariationalPosterior
        a copy of VariationalPosterior with the new traversed latents,
        the total number of return samples is: `n_samples * num * n_latents`

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
    """Conditioning the sampled dataset on known factors

    Parameters
    ----------
    known : Dict[Union[str, int], Callable[[int], bool]]
        a mapping from index or name of factor to a callable, the
        callable must return a list of boolean indices, which indicates
        the samples to be selected
    logical_not : bool, optional
        if True applying the opposed conditioning of the known factors, by default False
    n_samples : Optional[int], optional
        maximum number of selected samples, by default None
    seed : int, optional
        random seed for deterministic results, by default 1

    Returns
    -------
    VariationalPosterior
        a new posterior with conditioned factors

    Example
    -------
    ```
    # conditioning on: (1st-factor > 2) and (2nd-factor == 3)
    conditioning({1: lambda x: x > 2,
                    2: lambda x: x==3})
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
