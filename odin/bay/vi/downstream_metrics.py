from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict
from typing import List, Optional, Type, Union
from typing_extensions import Literal

import numpy as np
import scipy as sp
import tensorflow as tf
from numpy.random import RandomState
from odin.bay.distributions import Blockwise
from odin.bay.helpers import batch_slice
from odin.bay.vi.utils import discretizing
from odin.stats import is_discrete
from odin.utils import fifodict
from sklearn.base import BaseEstimator
from odin.ml.tree import fast_gbtree_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.distributions import Distribution
from tqdm import tqdm

__all__ = [
    'separated_attr_predictability',
    'importance_matrix',
    'dci_scores',
    'beta_vae_score',
    'factor_vae_score',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _to_numpy(x) -> np.ndarray:
  if isinstance(x, Distribution):
    x = x.mean()
  if hasattr(x, 'numpy'):
    x = x.numpy()
  return x


_cached_importance_matrix = fifodict(maxlen=10)


# ===========================================================================
# Disentanglement, completeness, informativeness
# ===========================================================================
def disentanglement_score(matrix):
  """Compute the disentanglement score of the representation.

  Arguments:
    matrix : is of shape `[n_latents, n_factors]`.
  """
  per_code = 1. - sp.stats.entropy(matrix + 1e-11, base=matrix.shape[1], axis=1)
  if matrix.sum() == 0.:
    matrix = np.ones_like(matrix)
  code_importance = matrix.sum(axis=1) / matrix.sum()
  return np.sum(per_code * code_importance)


def completeness_score(matrix):
  """Compute completeness of the representation.

  Arguments:
    matrix : is of shape `[n_latents, n_factors]`.
  """
  per_factor = 1. - sp.stats.entropy(
      matrix + 1e-11, base=matrix.shape[0], axis=0)
  if matrix.sum() == 0.:
    matrix = np.ones_like(matrix)
  factor_importance = matrix.sum(axis=0) / matrix.sum()
  return np.sum(per_factor * factor_importance)


def importance_matrix(
    repr_train: Union[Distribution, tf.Tensor, np.ndarray],
    factor_train: Union[tf.Tensor, np.ndarray],
    repr_test: Optional[Union[Distribution, tf.Tensor, np.ndarray]] = None,
    factor_test: Optional[Union[tf.Tensor, np.ndarray]] = None,
    test_size: float = 0.2,
    seed: int = 1,
    verbose: bool = False,
    n_jobs: Optional[int] = 4,
    cache_key: Optional[str] = None,
):
  """Using Tree Classifier to estimate the importance of each
  representation for each factor.

  Parameters
  -----------
  repr_train, repr_test : a Matrix `(n_samples, n_features)`
    input features for training the classifier
  factor_train, factor_test : a Matrix `(n_samples, n_factors)`
    discrete labels for the classifier

  Returns
  --------
  importance_matrix : a Matrix of shape `(n_features, n_factors)`
  train accuracy : a Scalar
  test accuracy : a Scalar
  """
  ## check the cahce
  if cache_key is not None and cache_key in _cached_importance_matrix:
    return _cached_importance_matrix[cache_key]
  ## preprocess data
  repr_train = _to_numpy(repr_train)
  if repr_test is not None:
    repr_test = _to_numpy(repr_test)
  n_latents = repr_train.shape[1]
  n_factors = factor_train.shape[1]
  ## split the datasets
  if repr_test is None or factor_test is None:
    repr_train, repr_test, factor_train, factor_test = train_test_split(
        repr_train, factor_train, test_size=0.2, random_state=seed)
  repr_train = np.asarray(repr_train)
  factor_train = np.asarray(factor_train)

  def _train(factor_idx):
    model = fast_gbtree_classifier(X=repr_train,
                                   y=factor_train[:, factor_idx],
                                   random_state=seed,
                                   n_jobs=n_jobs)
    feat = np.abs(model.feature_importances_)
    train = np.mean(model.predict(repr_train) == factor_train[:, factor_idx])
    test = np.mean(model.predict(repr_test) == factor_test[:, factor_idx])
    return factor_idx, feat, train, test

  # ====== compute importance based on gradient boosted trees ====== #
  matrix = np.zeros(shape=[n_latents, n_factors], dtype=np.float64)
  train_acc = list(range(n_factors))
  test_acc = list(range(n_factors))
  progress = list(range(n_factors))
  if verbose:
    progress = tqdm(progress, desc=f'Fitting GBT', unit='model')
  for factor_idx in progress:
    i, feat, train, test = _train(factor_idx)
    matrix[:, i] = feat
    train_acc[i] = train
    test_acc[i] = test
  if verbose:
    progress.clear()
    progress.close()
  rets = (matrix, train_acc, test_acc)
  if cache_key is not None:
    _cached_importance_matrix[cache_key] = rets
  return rets


def dci_scores(
    repr_train: Union[Distribution, tf.Tensor, np.ndarray],
    factor_train: Union[tf.Tensor, np.ndarray],
    repr_test: Optional[Union[Distribution, tf.Tensor, np.ndarray]] = None,
    factor_test: Optional[Union[tf.Tensor, np.ndarray]] = None,
    test_size: float = 0.2,
    seed: int = 1,
    verbose: bool = False,
    **kwargs,
):
  """Disentanglement, completeness, informativeness

  Parameteres
  ------------
  repr_train, repr_test : 2-D matrix `[n_samples, latent_dim]`
  factor_train, factor_test : 2-D matrix `[n_samples, n_factors]`

  Returns
  --------
  tuple of 3 scores (disentanglement, completeness, informativeness), all
    scores are higher is better.
    - disentanglement score: The degree to which a representation factorises
      or disentangles the underlying factors of variation
    - completeness score: The degree to which each underlying factor is
      captured by a single code variable.
    - informativeness score: test accuracy of a factor recognizer trained
      on train data

  References
  ------------
  Based on "A Framework for the Quantitative Evaluation of Disentangled
      Representations" (https://openreview.net/forum?id=By-7dz-AZ).

  Note
  -----
  This implementation only return accuracy on test data as informativeness
      score
  """
  importance, train_acc, test_acc = importance_matrix(repr_train,
                                                      factor_train,
                                                      repr_test,
                                                      factor_test,
                                                      seed=seed,
                                                      verbose=verbose,
                                                      **kwargs)
  train_acc = np.mean(train_acc)
  test_acc = np.mean(test_acc)
  # ====== disentanglement and completeness ====== #
  d = disentanglement_score(importance)
  c = completeness_score(importance)
  i = test_acc
  return d, c, i


def separated_attr_predictability(
    repr_train: Union[Distribution, tf.Tensor, np.ndarray],
    factor_train: Union[tf.Tensor, np.ndarray],
    repr_test: Optional[Union[Distribution, tf.Tensor, np.ndarray]] = None,
    factor_test: Optional[Union[tf.Tensor, np.ndarray]] = None,
    test_size: float = 0.2,
    continuous_factors: bool = False,
    seed: int = 1,
    max_iter: int = 2000,
):
  """The SAP score

  Parameters
  ----------
  repr_train, repr_test : `[n_samples, n_latents]`, the continuous
    latent representation.
  factor_train, factor_test : `[n_samples, n_factors]`. The groundtruth
    factors, could be continuous or discrete
  continuous_factors : A Boolean, indicate if the factor is discrete or
    continuous

  Reference
  ---------
  Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. Variational Inference of
      Disentangled Latent Concepts from Unlabeled Observations.
      arXiv:1711.00848 [cs, stat].
  """
  repr_train = _to_numpy(repr_train)
  if repr_test is not None:
    repr_test = _to_numpy(repr_test)
  n_latents = repr_train.shape[1]
  n_factors = factor_train.shape[1]
  ## split the datasets
  if repr_test is None or factor_test is None:
    repr_train, repr_test, factor_train, factor_test = train_test_split(
        repr_train, factor_train, test_size=0.2, random_state=seed)
  # ====== compute the score matrix ====== #
  score_matrix = np.zeros([n_latents, n_factors])
  for i in range(n_latents):
    for j in range(n_factors):
      x_i = repr_train[:, i]
      y_j = factor_train[:, j]
      if continuous_factors:
        # Attribute is considered continuous.
        cov_x_i_y_j = np.cov(x_i, y_j, ddof=1)
        cov_x_y = cov_x_i_y_j[0, 1]**2
        var_x = cov_x_i_y_j[0, 0]
        var_y = cov_x_i_y_j[1, 1]
        if var_x > 1e-12:
          score_matrix[i, j] = cov_x_y * 1. / (var_x * var_y)
        else:
          score_matrix[i, j] = 0.
      else:
        # Attribute is considered discrete.
        x_i_test = repr_test[:, i]
        y_j_test = factor_test[:, j]
        classifier = SVC(kernel='linear',
                         C=0.01,
                         max_iter=max_iter,
                         class_weight='balanced',
                         random_state=seed)
        # LinearSVC(C=0.01,
        #                        max_iter=max_iter,
        #                        class_weight="balanced",
        #                        random_state=seed)
        normalizer = StandardScaler()
        classifier.fit(normalizer.fit_transform(np.expand_dims(x_i, axis=-1)),
                       y_j)
        pred = classifier.predict(
            normalizer.transform(np.expand_dims(x_i_test, axis=-1)))
        score_matrix[i, j] = np.mean(pred == y_j_test)
        del classifier
  # ====== compute_avg_diff_top_two ====== #
  # [n_latents, n_factors]
  sorted_matrix = np.sort(score_matrix, axis=0)
  return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


# ===========================================================================
# SISUA method
# ===========================================================================
def predictive_strength(representations: tfd.Distribution,
                        factors: np.ndarray,
                        batch_size=8,
                        n_samples=1000):
  representations = tf.nest.flatten(representations)
  if len(representations) > 1:
    representations = Blockwise(representations)
  else:
    representations = representations[0]
  ### sampling
  exit()


# ===========================================================================
# betaVAE and factorVAE scoring methods
# ===========================================================================
def _sampling_helper(representations: tfd.Distribution,
                     factors: Union[tf.Tensor, np.ndarray],
                     rand: np.random.RandomState,
                     n_mcmc: int,
                     batch_size: int,
                     n_samples: int,
                     verbose: bool,
                     strategy: Literal['betavae', 'factorvae'],
                     desc: str = "Scoring",
                     **kwargs):
  assert isinstance(representations, tfd.Distribution),\
    f"representations must be instance of Distribution, but given: {type(representations)}"
  ## arguments
  size = representations.batch_shape[0]
  n_latents = representations.event_shape[0]
  n_factors = factors.shape[1]
  indices = np.arange(size, dtype=np.int64)
  ### create mapping factor -> list of representation_indices
  factor2ids = defaultdict(list)
  for idx, y in enumerate(factors.T):
    for sample_idx, i in enumerate(y):
      factor2ids[(idx, i)].append(sample_idx)
  ### prepare the output
  if strategy == 'betavae':
    features = np.empty(shape=(n_samples, n_latents), dtype=np.float32)
  elif strategy == 'factorvae':
    features = np.empty(shape=(n_samples,), dtype=np.int32)
    global_var = np.mean(representations.variance(), axis=0)
    # should > 0. here, otherwise, collapsed to prior
    # note: for Deterministic distribution variance = 0, hence, no active dims
    active_dims = np.sqrt(global_var) > 0.
  else:
    raise NotImplementedError(f"No support for sampling strategy: {strategy}")
  labels = np.empty(shape=(n_samples,), dtype=np.int32)
  ### data selector
  if n_mcmc == 0:  # use mean
    _X = representations.mean().numpy()
    get_x = lambda ids: _X[ids]
  else:
    _X = representations.sample(n_mcmc, seed=rand.randint(10e8)).numpy()
    # select the MCMC samples group then the data indices
    get_x = lambda ids: _X[rand.randint(0, n_mcmc)][ids]
  ### prepare the sampling progress
  if verbose:
    from tqdm import tqdm
    prog = tqdm(total=n_samples, desc=str(desc), unit='sample')
  count = 0
  while count < n_samples:
    factor_index = rand.randint(n_factors)
    ## betaVAE sampling
    if strategy == 'betavae':
      y = factors[rand.choice(indices, size=batch_size,
                              replace=True)][:, factor_index]
      obs1_ids = []
      obs2_ids = []
      for i in y:
        sample_indices = factor2ids[(factor_index, i)]
        if len(sample_indices) >= 2:
          s1, s2 = rand.choice(sample_indices, size=2, replace=False)
          obs1_ids.append(s1)
          obs2_ids.append(s2)
      # create the observation:
      if len(obs1_ids) > 0:
        obs1 = get_x(obs1_ids)
        obs2 = get_x(obs2_ids)
        feat = np.mean(np.abs(obs1 - obs2), axis=0)
        features[count, :] = feat
        labels[count] = factor_index
        count += 1
        if verbose:
          prog.update(1)
    ### factorVAE sampling
    elif strategy == 'factorvae':
      y = factors[rand.randint(size, dtype=np.int64), factor_index]
      obs_ids = factor2ids[(factor_index, y)]
      if len(obs_ids) > 1:
        obs_ids = rand.choice(obs_ids, size=batch_size, replace=True)
        obs = get_x(obs_ids)
        local_var = np.var(obs, axis=0, ddof=1)
        if not np.any(active_dims):  # no active dims
          features[count] = 0
        else:
          features[count] = np.argmin(local_var[active_dims] /
                                      global_var[active_dims])
        labels[count] = factor_index
        count += 1
        if verbose:
          prog.update(1)
  ## return shape: [n_samples, n_code] and [n_samples]
  if verbose:
    prog.clear()
    prog.close()
  return features, labels


def beta_vae_score(representations: tfd.Distribution,
                   factors: Union[tf.Tensor, np.ndarray],
                   n_mcmc: int = 10,
                   batch_size: int = 10,
                   n_samples: int = 10000,
                   seed: int = 1,
                   return_model: bool = False,
                   n_jobs: Optional[int] = None,
                   verbose: bool = False) -> float:
  """ The Beta-VAE score train a logistic regression to detect the invariant
  factor based on the absolute difference in the representations.

  References
  ----------
  beta-VAE: Learning Basic Visual Concepts with a Constrained
      Variational Framework (https://openreview.net/forum?id=Sy2fzU9gl).
  """
  desc = "betaVAE scoring"
  strategy = 'betavae'
  rand = RandomState(seed=seed)
  features, labels = _sampling_helper(**locals())
  ## train the classifier
  model = LogisticRegression(max_iter=2000,
                             random_state=rand.randint(1e8),
                             n_jobs=n_jobs)
  model.fit(features, labels)
  score = model.score(features, labels)
  if return_model:
    return score, model
  return score


def factor_vae_score(representations: tfd.Distribution,
                     factors: Union[tf.Tensor, np.ndarray],
                     n_mcmc: int = 10,
                     batch_size: int = 256,
                     n_samples: int = 10000,
                     seed: int = 1,
                     return_model: bool = False,
                     verbose: bool = False) -> float:
  """The Factor-VAE score train a highest-vote classifier to detect the
  invariant factor index from the lowest variated latent dimension.

  References
  ----------
  Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """
  desc = "factorVAE scoring"
  strategy = 'factorvae'
  rand = RandomState(seed=seed)
  features, labels = _sampling_helper(**locals())
  ## voting classifier
  n_latents = representations.event_shape[0]
  n_factors = factors.shape[1]
  votes = np.zeros((n_factors, n_latents), dtype=np.int64)
  for minvar_index, factor_index in zip(features, labels):
    votes[factor_index, minvar_index] += 1
  # factor labels for each latent code
  true_labels = np.argmax(votes, axis=0)
  # accuracy score
  score = np.sum(votes[true_labels, range(n_latents)]) / np.sum(votes)
  if return_model:
    return score, true_labels
  return score
