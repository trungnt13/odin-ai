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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
from odin.bay.vi.downstream_metrics import *
from odin.utils import catch_warnings_ignore
from odin.utils.mpi import MPI, get_cpu_count
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import completeness_score as _cluster_completeness_score
from sklearn.metrics import (homogeneity_score, mutual_info_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import entropy as entropy1D
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

__all__ = [
    'discrete_mutual_info',
    'discrete_entropy',
    'mutual_info_score',
    'mutual_info_estimate',
    'mutual_info_gap',
    'representative_importance_matrix',
    'dci_scores',
    # unsupervised scores
    'unsupervised_clustering_scores',
    # downstream score
    'separated_attr_predictability',
    'beta_vae_score',
    'factor_vae_score',
]


# ===========================================================================
# Clustering scores
# ===========================================================================
def _unsupervised_clustering_accuracy(y, y_pred):
  r""" Unsupervised Clustering Accuracy

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
  r""" Calculating the unsupervised clustering Scores:

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


def mutual_info_estimate(representations,
                         factors,
                         continuous_representations=True,
                         continuous_factors=False,
                         n_neighbors=3,
                         n_cpu=1,
                         seed=1):
  r""" Nonparametric method for estimating entropy from k-nearest neighbors
  distances (note: this implementation use multi-processing)

  Return:
    matrix `[num_latents, num_factors]`, estimated mutual information between
      each representation and each factors

  References:
    A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”.
      Phys. Rev. E 69, 2004.
    B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”.
      PLoS ONE 9(2), 2014.
    L. F. Kozachenko, N. N. Leonenko, “Sample Estimate of the Entropy of a
      Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16
  """
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

  jobs = list(range(num_factors))
  if n_cpu < 2:
    it = (func(i) for i in jobs)
  else:
    it = MPI(jobs=jobs, func=func, ncpu=n_cpu, batch=1)
  for i, mi in it:
    mi_matrix[:, i] = mi
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


# ===========================================================================
# Disentanglement, completeness, informativeness
# ===========================================================================
def disentanglement_score(importance_matrix):
  r""" Compute the disentanglement score of the representation.

  Arguments:
    importance_matrix : is of shape `[num_latents, num_factors]`.
  """
  per_code = 1. - sp.stats.entropy(
      importance_matrix + 1e-11, base=importance_matrix.shape[1], axis=1)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
  return np.sum(per_code * code_importance)


def completeness_score(importance_matrix):
  r""""Compute completeness of the representation.

  Arguments:
    importance_matrix : is of shape `[num_latents, num_factors]`.
  """
  per_factor = 1. - sp.stats.entropy(
      importance_matrix + 1e-11, base=importance_matrix.shape[0], axis=0)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor * factor_importance)


def representative_importance_matrix(repr_train,
                                     factor_train,
                                     repr_test,
                                     factor_test,
                                     random_state=1234,
                                     algo=GradientBoostingClassifier):
  r""" Using Tree Classifier to estimate the importance of each
  representation for each factor.

  Arguments:
    repr_train, repr_test : a Matrix `(n_samples, n_features)`
      input features for training the classifier
    factor_train, factor_test : a Matrix `(n_samples, n_factors)`
      discrete labels for the classifier
    algo : `sklearn.Estimator`, a classifier with `feature_importances_`
      attribute, for example:
        averaging methods:
        - `sklearn.ensemble.ExtraTreesClassifier`
        - `sklearn.ensemble.RandomForestClassifier`
        - `sklearn.ensemble.IsolationForest`
        and boosting methods:
        - `sklearn.ensemble.GradientBoostingClassifier`
        - `sklearn.ensemble.AdaBoostClassifier`

  Return:
    importance_matrix : a Matrix of shape `(n_features, n_factors)`
    train accuracy : a Scalar
    test accuracy : a Scalar
  """
  num_latents = repr_train.shape[1]
  num_factors = factor_train.shape[1]
  assert hasattr(algo, 'feature_importances_'), \
    "The class must contain 'feature_importances_' attribute"

  def _train(factor_idx):
    model = algo(random_state=random_state, n_iter_no_change=100)
    model.fit(np.asarray(repr_train), np.asarray(factor_train[:, factor_idx]))
    feat = np.abs(model.feature_importances_)
    train = np.mean(model.predict(repr_train) == factor_train[:, factor_idx])
    test = np.mean(model.predict(repr_test) == factor_test[:, factor_idx])
    return factor_idx, feat, train, test

  # ====== compute importance based on gradient boosted trees ====== #
  importance_matrix = np.zeros(shape=[num_latents, num_factors],
                               dtype=np.float64)
  train_acc = list(range(num_factors))
  test_acc = list(range(num_factors))
  ncpu = min(max(1, get_cpu_count() - 1), 10)
  for factor_idx in range(num_factors):
    i, feat, train, test = _train(factor_idx)
    importance_matrix[:, i] = feat
    train_acc[i] = train
    test_acc[i] = test
  return importance_matrix, train_acc, test_acc


def dci_scores(repr_train,
               factor_train,
               repr_test,
               factor_test,
               random_state=1234):
  r""" Disentanglement, completeness, informativeness

  Arguments:
    repr_train, repr_test : 2-D matrix `[n_samples, latent_dim]`
    factor_train, factor_test : 2-D matrix `[n_samples, n_factors]`

  Return:
    tuple of 3 scores (disentanglement, completeness, informativeness), all
      scores are higher is better.
      - disentanglement score: The degree to which a representation factorises
        or disentangles the underlying factors of variation
      - completeness score: The degree to which each underlying factor is
        captured by a single code variable.
      - informativeness score: test accuracy of a factor recognizer trained
        on train data

  References:
    Based on "A Framework for the Quantitative Evaluation of Disentangled
    Representations" (https://openreview.net/forum?id=By-7dz-AZ).

  Note:
    This implementation only return accuracy on test data as informativeness
      score
  """
  importance, train_acc, test_acc = representative_importance_matrix(
      repr_train, factor_train, repr_test, factor_test, random_state)
  train_acc = np.mean(train_acc)
  test_acc = np.mean(test_acc)
  # ====== disentanglement and completeness ====== #
  d = disentanglement_score(importance)
  c = completeness_score(importance)
  i = test_acc
  return d, c, i


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
