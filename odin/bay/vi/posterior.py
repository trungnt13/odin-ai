from __future__ import absolute_import, annotations, division, print_function

import random
import warnings
from collections import Counter, OrderedDict
from contextlib import contextmanager
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import ndarray
from odin import visual as vs
from odin.bay.distributions import CombinedDistribution
from odin.bay.vi._base import VariationalModel
from odin.bay.vi.metrics import (correlation_matrix, mutual_info_estimate,
                                 mutual_info_gap,
                                 representative_importance_matrix)
from odin.bay.vi.utils import discretizing, traverse_dims
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

__all__ = ['GroundTruth', 'VariationalPosterior']


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


def _boostrap_sampling(
    model: VariationalModel,
    inputs: List[ndarray],
    groundtruth: GroundTruth,
    n_samples: int,
    batch_size: int,
    verbose: bool,
    seed: int,
):
  from odin.bay.helpers import concat_distributions
  assert inputs.shape[0] == groundtruth.shape[0], \
    ('Number of samples mismatch between inputs and ground-truth, '
     f'{inputs.shape[0]} != {groundtruth.shape[0]}')
  inputs = as_tuple(inputs)
  Xs = [list() for _ in range(len(inputs))]  # inputs
  Zs = []  # latents
  Os = []  # outputs
  indices = []
  n = 0
  random_state = np.random.RandomState(seed=seed)
  prog = tqdm(desc=f'Sampling', total=n_samples, disable=not verbose)
  while n < n_samples:
    batch = min(batch_size, n_samples - n, groundtruth.shape[0])
    if verbose:
      prog.update(batch)
    # factors
    _, ids = groundtruth.sample_factors(num=batch,
                                        return_indices=True,
                                        seed=random_state.randint(0, 1e8))
    indices.append(ids)
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
    z = model.encode(inps[0] if len(inps) == 1 else inps, training=False)
    o = tf.nest.flatten(as_tuple(model.decode(z, training=False)))
    # post-process latents
    z = as_tuple(z)
    if len(z) == 1:
      z = z[0]
    Os.append(o)
    Zs.append(z)
    # update the counter
    n += len(ids)
  # end progress
  prog.clear()
  prog.close()
  # aggregate all data
  Xs = [np.concatenate(x, axis=0) for x in Xs]
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
  indices = np.concatenate(indices, axis=0)
  groundtruth = groundtruth[indices]
  return Xs, groundtruth, Zs, Os, indices


# ===========================================================================
# GroundTruth
# ===========================================================================
class GroundTruth:
  """Discrete factor for disentanglement analysis. If the factors is continuous,
  the values are casted to `int64` For discretizing continuous factor
  `odin.bay.vi.discretizing`

  Parameters
  ----------
  factors : [type]
      `[num_samples, n_factors]`, an Integer array
  factor_names : [type], optional
      None or `[n_factors]`, list of name for each factor, by default None
  categorical : Union[bool, List[bool]], optional
      list of boolean indicator if the given factor is categorical values or
      continuous values, this gives significant meaning when trying to visualize
      the factors, by default False

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

  def __init__(
      self,
      factors: Union[tf.Tensor, np.ndarray, DatasetV2],
      factor_names: Optional[List[str]] = None,
      categorical: Union[bool, List[bool]] = False,
      n_bins: Optional[Union[int, List[int]]] = None,
      strategy: Literal['uniform', 'quantile', 'kmeans', 'gmm'] = 'uniform',
  ):
    if isinstance(factors, tf.data.Dataset):
      factors = tf.stack([x for x in factors])
    if tf.is_tensor(factors):
      factors = factors.numpy()
    factors = np.atleast_2d(factors)
    if factors.ndim != 2:
      raise ValueError("factors must be a matrix [n_observations, n_factor], "
                       f"but given shape:{factors.shape}")
    # check factors is one-hot encoded
    if np.all(np.sum(factors, axis=-1) == 1):
      factors = np.argmax(factors, axis=1)[:, np.newaxis]
      categorical = True
    n_factors = factors.shape[1]
    # discretizing
    factors_original = np.array(factors)
    n_bins = as_tuple(n_bins, N=n_factors)
    strategy = as_tuple(strategy, N=n_factors, t=str)
    for i, (b, s) in enumerate(zip(n_bins, strategy)):
      if b is not None:
        factors[:, i] = discretizing(factors[:, i][:, np.newaxis],
                                     n_bins=b,
                                     strategy=s).ravel()
    factors = factors.astype(np.int64)
    # factor_names
    if factor_names is None:
      factor_names = [f'F{i}' for i in range(n_factors)]
    else:
      factor_names = [str(i) for i in tf.nest.flatten(factor_names)]
    assert len(factor_names) == n_factors, \
      f'Given {n_factors} but only {len(factor_names)} names'
    # store the attributes
    self.factors = factors
    self.factors_original = factors_original
    self.discretizer = list(zip(n_bins, strategy))
    self.categorical = as_tuple(categorical, N=n_factors, t=bool)
    self.names = factor_names
    self.labels = [np.unique(x) for x in factors.T]
    self.sizes = [len(lab) for lab in self.labels]

  def is_categorical(self, factor_index: Union[int, str]) -> bool:
    if isinstance(factor_index, string_types):
      factor_index = self.names.index(factor_index)
    return self.categorical[factor_index]

  def copy(self) -> GroundTruth:
    obj = GroundTruth.__new__(GroundTruth)
    obj.factors = self.factors
    obj.factors_original = self.factors_original
    obj.discretizer = self.discretizer
    obj.categorical = self.categorical
    obj.names = self.names
    obj.labels = self.labels
    obj.sizes = self.sizes
    return obj

  def __getitem__(self, key):
    obj = self.copy()
    obj.factors = obj.factors[key]
    obj.factors_original = obj.factors_original[key]
    return obj

  @property
  def shape(self) -> List[int]:
    return self.factors.shape

  @property
  def dtype(self) -> np.dtype:
    return self.factors.dtype

  @property
  def n_factors(self) -> int:
    return self.factors.shape[1]

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
      factors : `[num, n_factors]`
      indices (optional) : list of Integer
    """
    random_state = np.random.RandomState(seed=seed)
    if not isinstance(known, dict):
      known = dict(known)
    known = {
        self.names.index(k)
        if isinstance(k, string_types) else int(k): v \
          for k, v in known.items()
    }
    # make sure value of known factor is the actual label
    for idx, val in list(known.items()):
      labels = self.labels[idx]
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
      factors : `[num_samples, n_factors]`
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

  def __str__(self):
    text = f'GroundTruth: {self.factors.shape}\n'
    for i, (discretizer, name,
            labels) in enumerate(zip(self.discretizer, self.names,
                                     self.labels)):
      text += (f" [n={len(labels)}]'{name}'-"
               f"{discretizer}-"
               f"{'categorical' if self.categorical[i] else 'continuous'}: "
               f"{','.join([str(i) for i in labels])}\n")
    return text[:-1]


# ===========================================================================
# Sampler
# ===========================================================================
_CACHE_LATENTS = {}


class Posterior(vs.Visualizer):

  def __init__(self,
               model: keras.layers.Layer,
               groundtruth: GroundTruth,
               verbose: bool = False,
               name: str = 'Posterior',
               *args,
               **kwargs):
    super().__init__()
    assert isinstance(model, keras.layers.Layer), \
      f'model must be instance of keras.layers.Layer, but given:{type(model)}'
    assert isinstance(groundtruth, GroundTruth), \
      f'groundtruth must be instance of GroundTruth, but given:{type(groundtruth)}'
    self._name = str(name)
    self._model = model
    self._groundtruth = groundtruth
    self._dist_to_tensor = lambda d: d.sample()
    self._verbose = verbose

  @contextmanager
  def configure(
      self,
      dist_to_tensor: Optional[Callable[[Distribution], Tensor]] = None
  ) -> Posterior:
    d2t = self._dist_to_tensor
    if dist_to_tensor is not None:
      assert callable(dist_to_tensor), \
        ('fn must be a callable input a Distribution and return a Tensor, '
         f'given type:{dist_to_tensor}')
      self._dist_to_tensor = dist_to_tensor
    yield self
    self._dist_to_tensor = d2t

  def plot_scatter(
      self,
      factor_index: Union[int, str],
      classifier: Optional[Literal['svm', 'tree', 'logistic', 'knn', 'lda',
                                   'gbt']] = None,
      classifier_kw: Dict[str, Any] = {},
      dimension_reduction: Literal['pca', 'umap', 'tsne', 'knn',
                                   'kmean'] = 'tsne',
      max_samples: Optional[int] = 2000,
      return_figure: bool = False,
      ax: Optional['Axes'] = None,
      seed: int = 1,
  ) -> Union['Figure', Posterior]:
    """Plot dimension reduced scatter points of the sample set.

    Parameters
    ----------
    classifier : {'svm', 'tree', 'logistic', 'knn', 'lda', 'gbt'}, optional
        classifier for ploting decision contour of each factor, by default None
    classifier_kw : Dict[str, Any], optional
        keyword arguments for the classifier, by default {}
    dimension_reduction : {'pca', 'umap', 'tsne', 'knn', 'kmean'}, optional
        method for dimension reduction, by default 'tsne'
    factor_indices : Optional[Union[int, str, List[Union[int, str]]]], optional
        indicator of which factor will be plotted, by default None
    max_samples : Optional[int], optional
        maximum number of samples to be plotted, by default 2000
    return_figure : bool, optional
        return the figure or add it to the Visualizer for later processing,
        by default False
    seed : int, optional
        seed for random state, by default 1

    Returns
    -------
    Figure or Posterior
        return a `matplotlib.pyplot.Figure` if `return_figure=True` else return
        self for method chaining.
    """
    ## get all relevant factors
    if isinstance(factor_index, string_types):
      factor_index = self.factor_names.index(factor_index)
    factor_indices = int(factor_index)
    f = self.factors[:, factor_index]
    name = self.factor_names[factor_index]
    categorical = self.is_categorical(factor_index)
    f_norm = (f - np.mean(f, axis=0)) / np.std(f, axis=0)
    ## reduce latents dimension
    z = self.dimension_reduce(algorithm=dimension_reduction, seed=seed)
    x_min, x_max = np.min(z[:, 0]), np.max(z[:, 0])
    y_min, y_max = np.min(z[:, 1]), np.max(z[:, 1])
    ## downsample
    if isinstance(max_samples, Number):
      max_samples = int(max_samples)
      if max_samples < z.shape[0]:
        rand = np.random.RandomState(seed=seed)
        ids = rand.choice(np.arange(z.shape[0], dtype=np.int32),
                          size=max_samples,
                          replace=False)
        z = z[ids]
        f = f[ids]
    ## train classifier if provided
    n_samples = z.shape[0]
    if classifier is not None:
      xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_samples),
                           np.linspace(y_min, y_max, n_samples))
      xy = np.c_[xx.ravel(), yy.ravel()]
    ## plotting
    ax = vs.to_axis(ax, is_3D=False)
    cmap = 'bwr'
    # scatter plot
    vs.plot_scatter(x=z, val=f, color=cmap, ax=ax, size=10., alpha=0.5)
    ax.grid(False)
    ax.tick_params(axis='both',
                   bottom=False,
                   top=False,
                   left=False,
                   right=False)
    ax.set_title(name)
    # classifier boundary
    if classifier is not None:
      model = linear_classifier(z,
                                f,
                                algo=classifier,
                                seed=seed,
                                **classifier_kw)
      ax.contourf(xx,
                  yy,
                  model.predict(xy).reshape(xx.shape),
                  cmap=cmap,
                  alpha=0.4)
    if return_figure:
      return plt.gcf()
    return self.add_figure(
        name=f'scatter_{dimension_reduction}_{str(classifier).lower()}',
        fig=plt.gcf())

  def plot_histogram(
      self,
      histogram_bins: int = 120,
      original_factors: bool = True,
      return_figure: bool = False,
  ):
    Z = self.dist_to_tensor(self.latents).numpy()
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
        corr = self.mutualinfo_matrix(convert_to_tensor=self.dist_to_tensor,
                                      seed=seed)
        score_type = 'mutual-info'
      else:
        corr = self.correlation_matrix(convert_to_tensor=self.dist_to_tensor,
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
    Z = self.dist_to_tensor(self.latents).numpy()[:, latent_indices]
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
  def name(self) -> str:
    return self._name

  @property
  def model(self) -> keras.layers.Layer:
    return self._model

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
    return self._groundtruth.factors

  @property
  def factors_original(self) -> ndarray:
    return self._groundtruth.factors_original

  @property
  def factor_names(self) -> List[str]:
    return self._groundtruth.names

  @property
  def n_factors(self) -> int:
    return self.factors.shape[1]

  @property
  def n_samples(self) -> int:
    return self.factors.shape[0]

  def is_categorical(self, factor_index: Union[int, str]) -> bool:
    return self._groundtruth.is_categorical(factor_index)

  @property
  def latent_names(self) -> List[str]:
    raise NotImplementedError

  @property
  def n_latents(self) -> int:
    return self.latents.event_shape[0]

  @property
  def dist_to_tensor(self) -> Callable[[Distribution], Tensor]:
    return self._dist_to_tensor

  @property
  def verbose(self) -> bool:
    return self._verbose

  ############## Matrices
  def dimension_reduce(
      self,
      algorithm: Literal['pca', 'umap', 'tsne', 'knn', 'kmean'] = 'tsne',
      seed: int = 1,
  ) -> np.ndarray:
    """Applying dimension reduction on latents space, this method will cache the
    returns to lower computational cost."""
    key = f'{id(self)}_{id(self.dist_to_tensor)}_{algorithm}_{int(seed)}'
    if key in _CACHE_LATENTS:
      return _CACHE_LATENTS[key]
    x = self.dist_to_tensor(self.latents).numpy()
    x = dimension_reduce(x, algo=algorithm, random_state=seed)
    _CACHE_LATENTS[key] = x
    return x

  def correlation_matrix(
      self,
      method: Literal['spearman', 'pearson', 'lasso', 'average'] = 'spearman',
      sort_pairs: bool = False,
      seed: int = 1,
  ) -> ndarray:
    """Correlation matrix of `latent codes` (row) and `groundtruth factors`
    (column).

    Parameters
    ----------
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
    corr_mat = correlation_matrix(x1=self.dist_to_tensor(self.latents),
                                  x2=self.factors,
                                  method=method,
                                  seed=seed)
    ## decoding and return
    if sort_pairs:
      ids = diagonal_linear_assignment(corr_mat)
      corr_mat = corr_mat[ids, :]
      return corr_mat, OrderedDict(zip(range(self.n_factors), ids))
    return corr_mat

  def mutualinfo_matrix(
      self,
      n_neighbors: Union[int, List[int]] = [3, 4, 5],
      n_cpu: int = 1,
      seed: int = 1,
  ) -> np.ndarray:
    """Mutual Information estimated between each latents' dimension and factors'
    dimension

    Parameters
    ----------
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
            representations=self.dist_to_tensor(self.latents).numpy(),
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
      n_bins: int = 10,
      strategy: Literal['uniform', 'quantile', 'kmeans', 'gmm'] = 'uniform',
  ) -> float:
    """Mutual Information Gap

    Parameters
    ----------
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
    z = self.dist_to_tensor(self.latents).numpy()
    if n_bins > 1:
      z = discretizing(z, independent=True, n_bins=n_bins, strategy=strategy)
    f = self.factors
    return mutual_info_gap(z, f)

  def copy(self, suffix='copy') -> Posterior:
    obj = self.__class__.__new__(self.__class__)
    obj._name = f'{self.name}_{suffix}'
    obj._verbose = self._verbose
    obj._model = self._model
    obj._groundtruth = self._groundtruth
    obj._dist_to_tensor = self._dist_to_tensor
    return obj


# ===========================================================================
# Variational Posterior
# ===========================================================================
class VariationalPosterior(Posterior):
  """Posterior class for variational inference using Variational Autoencoder"""

  def __init__(self,
               model: VariationalModel,
               groundtruth: GroundTruth,
               inputs: Optional[Union[ndarray, Tensor]] = None,
               latents: Optional[Union[ndarray, Tensor, Distribution]] = None,
               n_samples: int = 5000,
               batch_size: int = 32,
               seed: int = 1,
               **kwargs):
    super().__init__(model=model, groundtruth=groundtruth, **kwargs)
    assert isinstance(model, VariationalModel), \
      ("model must be instance of odin.bay.vi.VariationalModel, "
       f"given: {type(model)}")
    ### prepare the inputs - latents
    if inputs is None and latents is None:
      raise ValueError("Either inputs or latents must be provided")
    ## latents are given directly
    if inputs is None:
      if isinstance(latents, (np.ndarray, tf.Tensor)):
        latents = VectorDeterministic(loc=latents, name='Latents')
      latents = as_tuple(latents)
      latents = latents[0] if len(latents) == 1 else \
        CombinedDistribution(latents, name="Latents")
      assert latents.batch_shape[0] == self.n_samples, \
        ('Number of samples mismatch between latents distribution and '
         f'ground-truth factors, {latents.batch_shape[0]} != {self.n_samples}')
      outputs = self.model.decode(latents, training=False)
      indices = None
    ## sampling the latents
    else:
      inputs, groundtruth, latents, outputs, indices = \
          _boostrap_sampling(self.model,
                         inputs=inputs,
                         groundtruth=self._groundtruth,
                         batch_size=batch_size,
                         n_samples=n_samples,
                         verbose=self.verbose,
                         seed=seed)
    ## assign the attributes
    self._inputs = inputs
    self._groundtruth = groundtruth
    self._latents = latents
    self._outputs = outputs
    self._indices = indices

  @property
  def model(self) -> VariationalModel:
    return self._model

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
  def latent_names(self) -> List[str]:
    return [f"Z{i}" for i in range(self.n_latents)]

  ############## Experiment setup
  def traverse(
      self,
      feature_indices: Union[int, List[int]],
      min_val: int = -2.0,
      max_val: int = 2.0,
      n_traverse_points: int = 11,
      n_random_samples: int = 1,
      mode: Literal['linear', 'quantile', 'gaussian'] = 'linear',
      seed: int = 1,
  ) -> VariationalPosterior:
    """Create data for latents' traverse experiments

    Parameters
    ----------
    feature_indices : Union[int, List[int]]
    min_val : int, optional
        minimum value of the traverse, by default -2.0
    max_val : int, optional
        maximum value of the traverse, by default 2.0
    n_traverse_points : int, optional
        number of points in the traverse, must be odd number, by default 11
    n_random_samples : int, optional
        number of samples selected for the traverse, by default 2
    mode : {'linear', 'quantile', 'gaussian'}, optional
        'linear' mode take linear interpolation between the `min_val` and `max_val`.
        'quantile' mode return `num` quantiles based on min and max values inferred
        from the data. 'gaussian' mode takes `num` Gaussian quantiles,
        by default 'linear'

    Returns
    -------
    VariationalPosterior
        a copy of VariationalPosterior with the new traversed latents,
        the total number of return samples is: `n_samples * num`

    Example
    --------
    For `n_samples=2`, `num=2`, and `n_latents=2`, the return latents are:
    ```
    [[-2., 0.47],
     [ 0., 0.47],
     [ 2., 0.47],
     [-2., 0.31],
     [ 0., 0.31],
     [ 2., 0.31]]
    ```
    """
    Z, indices = traverse_dims(x=self.latents.sample(seed=seed),
                               feature_indices=feature_indices,
                               min_val=min_val,
                               max_val=max_val,
                               n_traverse_points=n_traverse_points,
                               n_random_samples=n_random_samples,
                               mode=mode,
                               return_indices=True,
                               seed=seed)
    ### create the new posterior
    # NOTE: this might not work for multi-latents
    outputs = list(as_tuple(self.model.decode(Z, training=False)))
    obj = self.copy(indices,
                    latents=VectorDeterministic(Z, name="Latents"),
                    outputs=outputs,
                    suffix='traverse')
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
    return self.copy(ids, suffix='conditioning')

  def copy(self,
           indices: Optional[Union[slice, List[int]]] = None,
           latents: Optional[Distribution] = None,
           outputs: Optional[List[Distribution]] = None,
           suffix: str = 'copy') -> VariationalPosterior:
    """Return the deepcopy"""
    obj = super().copy(suffix=suffix)
    # helper for slicing
    fslice = lambda x: x[indices] if indices is not None else x
    # copy the factors
    obj._groundtruth = fslice(self._groundtruth.copy())
    # copy the inputs
    obj._inputs = [np.array(fslice(i)) for i in self._inputs]
    obj._indices = np.array(fslice(self._indices))
    ## inference for the latents and outputs
    if indices is not None:
      if latents is None:
        inputs = obj.inputs
        latents = self.model.encode(inputs[0] if len(inputs) == 1 else inputs,
                                    training=False)
      if outputs is None:
        outputs = self.model.decode(latents, training=False)
      latents = as_tuple(latents)
      obj._latents = CombinedDistribution(latents, name='Latents') \
        if len(latents) > 1 else latents[0]
      obj._outputs = list(as_tuple(outputs))
    ## just copy paste
    else:
      obj._latents = self._latents.copy()
      obj._outputs = [o.copy() for o in self._outputs]
    return obj

  def __str__(self):
    dname = lambda d: str(d).replace('Independent',
                                     d.distribution.__class__.__name__) \
      if isinstance(d, Independent) else str(d)
    return \
f"""{self.name}:
  model  : {self._model.__class__}
  dist2tensor: {self.dist_to_tensor}
  verbose: {self._verbose}
  factors: {self.factors.shape} - {', '.join(self.factor_names)}
  inputs : {', '.join(str((i.shape, i.dtype)) for i in self.inputs)}
  outputs: {', '.join(dname(o).replace('tfp.distributions.', '') for o in self.outputs)}
  latents: {dname(self.latents).replace('tfp.distributions.', '')}"""
