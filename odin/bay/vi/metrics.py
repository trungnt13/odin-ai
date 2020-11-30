# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
# (https://github.com/google-research/disentanglement_lib)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import warnings
from collections import defaultdict
from enum import IntFlag, auto
from functools import partial
from typing import Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf
import numpy as np
import scipy as sp
from odin.bay.vi.downstream_metrics import *
from odin.utils import catch_warnings_ignore
from odin.utils.mpi import MPI, get_cpu_count
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import completeness_score as _cluster_completeness_score
from sklearn.metrics import (homogeneity_score, mutual_info_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import entropy as entropy1D
from sklearn.mixture import GaussianMixture
from tensorflow_probability.python.distributions import Distribution
from tqdm import tqdm
from typeguard import typechecked
from typing_extensions import Literal
from odin.bay.vi.downstream_metrics import *

__all__ = [
    'correlation_matrix',
    'discrete_mutual_info',
    'discrete_entropy',
    'mutual_info_estimate',
    'mutual_info_gap',
    'relative_strength',
    # unsupervised scores
    'unsupervised_clustering_scores',
    'Correlation',
]

_cached_correlation_matrix = defaultdict(dict)
_cached_mi_matrix = {}

# ===========================================================================
# Correlation
# ===========================================================================
_corr_methods = ['spearman', 'lasso', 'pearson', 'average']


def correlation_matrix(
    x1: Union[np.ndarray, tf.Tensor],
    x2: Union[np.ndarray, tf.Tensor],
    method: Literal['spearman', 'pearson', 'lasso', 'average'] = 'spearman',
    seed: int = 1,
    cache_key: Optional[str] = None,
) -> np.ndarray:
  """Correlation matrix of each column in `x1` to each column in `x2`

  Parameters
  ----------
  x1 : np.ndarray
    a matrix
  x2 : np.ndarray
    a matrix, satisfying `x1.shape[0] == x2.shape[0]`
  method : {'spearman', 'pearson', 'lasso', 'average'}
      method for calculating the correlation,
      'spearman' - rank or monotonic correlation
      'pearson' - linear correlation
      'lasso' - lasso regression
      'average' - compute all known method then taking average,
      by default 'spearman'
  seed : int, optional
      random state seed, by default 1

  Returns
  -------
  ndarray
      correlation matrices `[x1.shape[1], x2.shape[1]]`, all entries are
      in `[0, 1]`.
  """
  if (cache_key is not None and
      cache_key in _cached_correlation_matrix[method]):
    return _cached_correlation_matrix[method][cache_key]
  x1 = np.asarray(x1)
  x2 = np.asarray(x2)
  d1 = x1.shape[-1]
  d2 = x2.shape[-1]
  method = str(method).strip().lower()
  assert x1.shape[0] == x2.shape[0], \
    f'Number of samples in x1 and x2 mismatch, {x1.shape[0]} and {x2.shape[0]}'
  assert method in _corr_methods, \
    f"Support {_corr_methods} correlation but given method='{method}'"
  ### average mode
  if method == 'average':
    corr_mat = sum(
        correlation_matrix(x1=x1, x2=x2, method=corr, seed=seed)
        for corr in ['spearman', 'pearson', 'lasso']) / 3
  ### specific mode
  else:
    # lasso
    if method == 'lasso':
      model = Lasso(random_state=seed, alpha=0.1)
      model.fit(x1, x2)
      # coef_ is [n_target, n_features], so we need transpose here
      corr_mat = np.transpose(np.absolute(model.coef_))
    # spearman and pearson
    else:
      corr_mat = np.empty(shape=(d1, d2), dtype=np.float64)
      for i1 in range(d1):
        for i2 in range(d2):
          j1, j2 = x1[:, i1], x2[:, i2]
          if method == 'spearman':
            corr = sp.stats.spearmanr(j1, j2, nan_policy="omit")[0]
          elif method == 'pearson':
            corr = sp.stats.pearsonr(j1, j2)[0]
          corr_mat[i1, i2] = corr
  ## decoding and return
  if cache_key is not None:
    _cached_correlation_matrix[method][cache_key] = corr_mat
  return corr_mat


# ===========================================================================
# Clustering scores
# ===========================================================================
def _unsupervised_clustering_accuracy(y, y_pred):
  """Unsupervised Clustering Accuracy

  Author: scVI (https://github.com/YosefLab/scVI)
  """
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert len(y_pred) == len(y)
  u = np.unique(np.concatenate((y, y_pred)))
  n_clusters = len(u)
  mapping = dict(zip(u, range(n_clusters)))
  reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
  for y_pred_, y_ in zip(y_pred, y):
    if y_ in mapping:
      reward_matrix[mapping[y_pred_], mapping[y_]] += 1
  cost_matrix = reward_matrix.max() - reward_matrix
  ind = linear_assignment(cost_matrix)
  return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind


def _clustering_scores(y, X=None, z=None, algo='kmeans', random_state=1):
  n_factors = len(np.unique(y))
  if z is None:
    if algo == 'kmeans':
      model = KMeans(n_factors, n_init=200, random_state=random_state)
    elif algo == 'gmm':
      model = GaussianMixture(n_factors, random_state=random_state)
    elif algo in ('both', 'avg', 'avr', 'average', 'mean'):
      score1 = _clustering_scores(X=X,
                                  y=y,
                                  z=z,
                                  algo='kmeans',
                                  random_state=random_state)
      score2 = _clustering_scores(X=X,
                                  y=y,
                                  z=z,
                                  algo='gmm',
                                  random_state=random_state)
      return {k: (v + score2[k]) / 2 for k, v in score1.items()}
    else:
      raise ValueError("Not support for prediction_algorithm: '%s'" % algo)
    # the scores
    y_pred = model.fit_predict(X)
  else:
    z = z.ravel()
    assert z.shape[0] == y.shape[0], \
      f"predictions must have shape: {y.shape}, but given: {z.shape}"
    y_pred = z
  with catch_warnings_ignore(FutureWarning):
    return dict(
        ASW=silhouette_score(X if X is not None else np.expand_dims(z, axis=-1),
                             y),
        ARI=adjusted_rand_score(y, y_pred),
        NMI=normalized_mutual_info_score(y, y_pred),
        UCA=_unsupervised_clustering_accuracy(y, y_pred)[0],
        HOS=homogeneity_score(y, y_pred),
        COS=_cluster_completeness_score(y, y_pred),
    )


def unsupervised_clustering_scores(factors: np.ndarray,
                                   representations: Optional[np.ndarray] = None,
                                   predictions: Optional[np.ndarray] = None,
                                   algorithm: str = 'both',
                                   random_state: int = 1,
                                   n_cpu: int = 1,
                                   verbose: bool = True) -> Dict[str, float]:
  """ Calculating the unsupervised clustering Scores:

    - ASW : silhouette_score ([-1, 1], higher is better)
        is calculated using the mean intra-cluster distance and the
        mean nearest-cluster distance (b) for each sample. Values near 0
        indicate overlapping clusters
    - ARI : adjusted_rand_score ([-1, 1], higher is better)
        A similarity measure between two clusterings by considering all pairs
        of samples and counting pairs that are assigned in the same or
        different clusters in the predicted and true clusterings.
        Similarity score between -1.0 and 1.0. Random labelings have an ARI
        close to 0.0. 1.0 stands for perfect match.
    - NMI : normalized_mutual_info_score ([0, 1], higher is better)
        Normalized Mutual Information between two clusterings.
        1.0 stands for perfectly complete labeling
    - UCA : unsupervised_clustering_accuracy ([0, 1], higher is better)
        accuracy of the linear assignment between predicted labels and
        ground-truth labels.
    - HOS : homogeneity_score ([0, 1], higher is better)
        A clustering result satisfies homogeneity if all of its clusters
        contain only data points which are members of a single class.
        1.0 stands for perfectly homogeneous
    - COS : completeness_score ([0, 1], higher is better)
        A clustering result satisfies completeness if all the data points
        that are members of a given class are elements of the same cluster.
        1.0 stands for perfectly complete labeling

  Arguments:
    factors : a Matrix.
      Categorical factors (i.e. one-hot encoded), or multiple factors.
    algorithm : {'kmeans', 'gmm', 'both'}.
      The clustering algorithm for assigning the cluster from representations

  Return:
    Dict mapping score alias to its scalar value

  Note:
    The time complexity is exponential as the number of labels increasing
  """
  if factors.ndim == 1:
    factors = np.expand_dims(factors, axis=-1)
  assert representations is not None or predictions is not None, \
    "either representations or predictions must be provided"
  ### preprocessing factors
  # multinomial :
  # binary :
  # multibinary :
  factor_type = 'multinomial'
  if np.all(np.unique(factors) == [0., 1.]):
    if np.all(np.sum(factors, axis=1) == 1.):
      factor_type = 'binary'
    else:
      factor_type = 'multibinary'
  # start scoring
  if factor_type == 'binary':
    return _clustering_scores(X=representations,
                              z=predictions,
                              y=np.argmax(factors, axis=1),
                              algo=algorithm,
                              random_state=random_state)
  if factor_type in ('multinomial', 'multibinary'):

    def _get_scores(idx):
      y = factors[:, idx]
      if factor_type == 'multinomial':
        uni = {v: i for i, v in enumerate(sorted(np.unique(y)))}
        y = np.array([uni[i] for i in y])
      else:
        y = y.astype(np.int32)
      return _clustering_scores(X=representations,
                                z=predictions,
                                y=y,
                                algo=algorithm,
                                random_state=random_state)

    scores = defaultdict(list)
    if factors.shape[1] == 1:
      verbose = False
    prog = tqdm(desc="Scoring clusters",
                total=factors.shape[1],
                disable=not verbose)
    if n_cpu == 1:
      it = (_get_scores(idx) for idx in range(factors.shape[1]))
    else:
      it = MPI(jobs=list(range(factors.shape[1])),
               func=_get_scores,
               batch=1,
               ncpu=n_cpu)
    for s in it:
      prog.update(1)
      for k, v in s.items():
        scores[k].append(v)
    return {k: np.mean(v) for k, v in scores.items()}


# ===========================================================================
# Mutual information
# ===========================================================================
def discrete_mutual_info(codes, factors):
  r"""Compute discrete mutual information.

  Arguments:
    codes : `[n_samples, n_codes]`, the latent codes or predictive codes
    factors : `[n_samples, n_factors]`, the groundtruth factors

  Return:
    matrix `[n_codes, n_factors]` : mutual information score between factor
      and code
  """
  codes = np.atleast_2d(codes)
  factors = np.atleast_2d(factors)
  assert codes.ndim == 2 and factors.ndim == 2, \
    "codes and factors must be matrix, but given: %s and %s" % \
      (str(codes.shape), str(factors.shape))
  num_latents = codes.shape[1]
  num_factors = factors.shape[1]
  m = np.zeros([num_latents, num_factors])
  for i in range(num_latents):
    for j in range(num_factors):
      m[i, j] = mutual_info_score(factors[:, j], codes[:, i])
  return m


def discrete_entropy(labels):
  r""" Iterately compute discrete entropy for integer samples set along the
  column of 2-D array.

  Arguments:
    labels : 1-D or 2-D array

  Returns:
    entropy : A Scalar or array `[n_factors]`
  """
  labels = np.atleast_1d(labels)
  if labels.ndim == 1:
    return entropy1D(labels.ravel())
  elif labels.ndim > 2:
    raise ValueError("Only support 1-D or 2-D array for labels entropy.")
  num_factors = labels.shape[1]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = entropy1D(labels[:, j])
  return h


def mutual_info_estimate(
    representations: np.ndarray,
    factors: np.ndarray,
    continuous_representations: bool = True,
    continuous_factors: bool = False,
    n_neighbors: int = 3,
    n_cpu: int = 1,
    seed: int = 1,
    verbose: bool = False,
    cache_key: Optional[str] = None,
) -> np.ndarray:
  r""" Nonparametric method for estimating entropy from k-nearest neighbors
  distances (note: this implementation use multi-processing)

  Parameters
  -----------

  Return
  --------
  matrix `[num_latents, num_factors]`, estimated mutual information between
    each representation and each factors

  References
  ------------
  A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”.
    Phys. Rev. E 69, 2004.
  B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”.
    PLoS ONE 9(2), 2014.
  L. F. Kozachenko, N. N. Leonenko, “Sample Estimate of the Entropy of a
    Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16
  """
  if cache_key is not None and cache_key in _cached_mi_matrix:
    return _cached_mi_matrix[cache_key]
  from sklearn.feature_selection import (mutual_info_classif,
                                         mutual_info_regression)
  mutual_info = mutual_info_regression if continuous_factors else \
    mutual_info_classif
  num_latents = representations.shape[1]
  num_factors = factors.shape[1]
  # iterate over each factor
  mi_matrix = np.empty(shape=(num_latents, num_factors), dtype=np.float64)

  # repeat for each factor
  def func(idx):
    mi = mutual_info(representations,
                     factors[:, idx],
                     discrete_features=not continuous_representations,
                     n_neighbors=n_neighbors,
                     random_state=seed)
    return idx, mi

  ## compute the MI matrix
  jobs = list(range(num_factors))
  if n_cpu < 2:
    it = (func(i) for i in jobs)
  else:
    it = MPI(jobs=jobs, func=func, ncpu=n_cpu, batch=1)
  if verbose:
    from tqdm import tqdm
    it = tqdm(it, desc='Estimating mutual information', total=len(jobs))
  for i, mi in it:
    mi_matrix[:, i] = mi
  ## return
  if cache_key is not None:
    _cached_mi_matrix[cache_key] = mi_matrix
  return mi_matrix


def mutual_info_gap(representations, factors):
  r"""Computes score based on both representation codes and factors.
    In (Chen et. al 2019), 10000 samples used to estimate MIG

  Arguments:
    representation : `[n_samples, n_latents]`, discretized latent
      representation
    factors : `[n_samples, n_factors]`, discrete groundtruth factor

  Return:
    A scalar: discrete mutual information gap score

  Reference:
    Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. Isolating Sources of
      Disentanglement in Variational Autoencoders. arXiv:1802.04942 [cs, stat].

  """
  representations = np.atleast_2d(representations).astype(np.int64)
  factors = np.atleast_2d(factors).astype(np.int64)
  # m is [n_latents, n_factors]
  m = discrete_mutual_info(representations, factors)
  sorted_m = np.sort(m, axis=0)[::-1]
  entropy_ = discrete_entropy(factors)
  return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy_[:]))


def relative_strength(mat):
  r""" Computes relative strength score for both axes of a correlation matrix.

  Arguments:
    mat : a Matrix. Correlation matrix with values range from -1 to 1.
  """
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    score_x = np.mean(np.nan_to_num(\
      np.power(np.max(mat, axis=0), 2) / np.sum(mat, axis=0),
      copy=False, nan=0.0))
    score_y = np.mean(np.nan_to_num(\
      np.power(np.max(mat, axis=1), 2) / np.sum(mat, axis=1),
      copy=False, nan=0.0))
  return (score_x + score_y) / 2


# ===========================================================================
# Summary
# ===========================================================================
class Correlation(IntFlag):
  """Generalized interface for estimating the correlation of two matrices:

  Parameters
  ----------
  x1 : Union[np.ndarray, tf.Tensor]
      representation matrix of shape `[n_samples, n_latents]`
  x2 : Union[np.ndarray, tf.Tensor]
      factor matrix of shape `[n_samples, n_factors]`
  seed : int, optional
      random seed, by default 1
  **kwargs : extra keywords arguments for the method

  Returns
  -------
  Union[np.ndarray, List[np.ndarray]]
      The output is correlation matrix of shape `[n_latents, n_factors]`

  """
  Pearson = auto()
  Spearman = auto()
  Lasso = auto()
  MutualInfo = auto()
  Importance = auto()

  def __iter__(self):
    for method in Correlation:
      if method in self:
        yield method

  def __len__(self):
    return len(list(iter(self)))

  @property
  def is_single(self) -> bool:
    return len(self) == 1

  def __call__(self,
               x1: Union[Distribution, np.ndarray, tf.Tensor],
               x2: Union[Distribution, np.ndarray, tf.Tensor],
               seed: int = 1,
               **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    if hasattr(x1, 'numpy'):
      x1 = x1.numpy()
    if hasattr(x2, 'numpy'):
      x2 = x2.numpy()
    if len(self) != 1:
      return [method(x1, x2, seed=seed, **kwargs) for method in self]
    if self == Correlation.Pearson:
      fn = partial(correlation_matrix, method='pearson')
    elif self == Correlation.Spearman:
      fn = partial(correlation_matrix, method='spearman')
    elif self == Correlation.Lasso:
      fn = partial(correlation_matrix, method='lasso')
    elif self == Correlation.MutualInfo:
      fn = mutual_info_estimate
    elif self == Correlation.Importance:
      fn = lambda *args, **kw: importance_matrix(*args, **kw)[0]
    return fn(x1, x2, seed=seed, **kwargs)
