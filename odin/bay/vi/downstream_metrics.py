from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from tensorflow_probability import distributions as tfd


def separated_attr_predictability(repr_train,
                                  factor_train,
                                  repr_test,
                                  factor_test,
                                  continuous_factors=False,
                                  random_state=1234):
  r""" The SAP score

  Arguments:
    repr_train, repr_test : `[n_samples, n_latents]`, the continuous
      latent representation.
    factor_train, factor_test : `[n_samples, n_factors]`. The groundtruth
      factors, could be continuous or discrete
    continuous_factors : A Boolean, indicate if the factor is discrete or
      continuous

  Reference:
    Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. Variational Inference of
      Disentangled Latent Concepts from Unlabeled Observations.
      arXiv:1711.00848 [cs, stat].

  """
  num_latents = repr_train.shape[1]
  num_factors = factor_train.shape[1]
  # ====== compute the score matrix ====== #
  score_matrix = np.zeros([num_latents, num_factors])
  for i in range(num_latents):
    for j in range(num_factors):
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
        classifier = LinearSVC(C=0.01,
                               max_iter=8000,
                               class_weight="balanced",
                               random_state=random_state)
        classifier.fit(np.expand_dims(x_i, axis=-1), y_j)
        pred = classifier.predict(np.expand_dims(x_i_test, axis=-1))
        score_matrix[i, j] = np.mean(pred == y_j_test)
  # ====== compute_avg_diff_top_two ====== #
  # [num_latents, num_factors]
  sorted_matrix = np.sort(score_matrix, axis=0)
  return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


# ===========================================================================
# BetaVAE and FactorVAE scoring methods
# ===========================================================================
def _sampling_helper(representations,
                     factors,
                     rand,
                     use_mean,
                     batch_size,
                     n_samples,
                     verbose,
                     strategy,
                     desc="Scoring",
                     **kwargs):
  assert isinstance(representations, tfd.Distribution),\
    "representations must be instance of Distribution, but given: %s" % \
      str(type(representations))
  ## arguments
  from odin.bay.distributions import slice_distribution
  size = representations.batch_shape[0]
  n_codes = representations.event_shape[0]
  n_factors = factors.shape[1]
  indices = np.arange(size, dtype=np.int64)
  ## create mapping factor -> representation_index
  code_map = defaultdict(list)
  for idx, y in enumerate(factors.T):
    for sample_idx, i in enumerate(y):
      code_map[(idx, i)].append(sample_idx)
  ## prepare the output
  if strategy == 'betavae':
    features = np.empty(shape=(n_samples, n_codes), dtype=np.float32)
  elif strategy == 'factorvae':
    features = np.empty(shape=(n_samples,), dtype=np.int32)
    global_var = np.mean(representations.variance(), axis=0)
    # should > 0. here, otherwise, collapsed to prior
    active_dims = np.sqrt(global_var) > 0.
  else:
    raise NotImplementedError("No support for sampling strategy: %s" % strategy)
  labels = np.empty(shape=(n_samples,), dtype=np.int32)
  repr_fn = (lambda d: d.mean()) if use_mean else (lambda d: d.sample())
  count = 0
  ## prepare the sampling progress
  if verbose:
    from tqdm import tqdm
    prog = tqdm(total=n_samples, desc=str(desc), unit='sample')
  while count < n_samples:
    factor_index = rand.randint(n_factors)
    ## betaVAE sampling
    if strategy == 'betavae':
      y = factors[rand.choice(indices, size=batch_size,
                              replace=True)][:, factor_index]
      obs1_ids = []
      obs2_ids = []
      for i in y:
        sample_indices = code_map[(factor_index, i)]
        if len(sample_indices) >= 2:
          s1, s2 = rand.choice(sample_indices, size=2, replace=False)
          obs1_ids.append(s1)
          obs2_ids.append(s2)
      # create the observation:
      if len(obs1_ids) > 0:
        obs1 = slice_distribution(obs1_ids, representations)
        obs2 = slice_distribution(obs2_ids, representations)
        feat = np.mean(np.abs(repr_fn(obs1) - repr_fn(obs2)), axis=0)
        features[count, :] = feat
        labels[count] = factor_index
        count += 1
        if verbose:
          prog.update(1)
    ## factorVAE sampling
    elif strategy == 'factorvae':
      y = factors[rand.randint(size, dtype=np.int64), factor_index]
      obs_ids = code_map[(factor_index, y)]
      if len(obs_ids) > 1:
        obs_ids = rand.choice(obs_ids, size=batch_size, replace=True)
        obs = slice_distribution(obs_ids, representations)
        local_var = np.var(repr_fn(obs), axis=0, ddof=1)
        features[count] = np.argmin(local_var[active_dims] /
                                    global_var[active_dims])
        labels[count] = factor_index
        count += 1
        if verbose:
          prog.update(1)
  ## return shape: [n_samples, n_code] and [n_samples]
  return features, labels


def beta_vae_score(representations: tfd.Distribution,
                   factors: np.ndarray,
                   use_mean=False,
                   batch_size=8,
                   n_samples=1000,
                   random_state=1234,
                   return_model=False,
                   verbose=False):
  r""" The Beta-VAE score train a logistic regression to detect the invariant
  factor based on the absolute difference in the representations.

  References:
    beta-VAE: Learning Basic Visual Concepts with a Constrained
      Variational Framework (https://openreview.net/forum?id=Sy2fzU9gl).

  """
  desc = "BetaVAE scoring"
  strategy = 'betavae'
  rand = random_state if isinstance(random_state, RandomState) else \
    RandomState(seed=random_state)
  features, labels = _sampling_helper(**locals())
  ## train the classifier
  model = LogisticRegression(max_iter=1000, random_state=rand.randint(1e8))
  model.fit(features, labels)
  score = model.score(features, labels)
  if return_model:
    return score, model
  return score


def factor_vae_score(representations: tfd.Distribution,
                     factors: np.ndarray,
                     use_mean=False,
                     batch_size=8,
                     n_samples=1000,
                     random_state=1234,
                     return_model=False,
                     verbose=False):
  r""" The Factor-VAE score train a highest-vote classifier to detect the
  invariant factor index from the lowest variated latent dimension.

  References:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """
  desc = "FactorVAE scoring"
  strategy = 'factorvae'
  rand = random_state if isinstance(random_state, RandomState) else \
    RandomState(seed=random_state)
  features, labels = _sampling_helper(**locals())
  ## voting classifier
  n_codes = representations.event_shape[0]
  n_factors = factors.shape[1]
  votes = np.zeros((n_factors, n_codes), dtype=np.int64)
  for minvar_index, factor_index in zip(features, labels):
    votes[factor_index, minvar_index] += 1
  # factor labels for each latent code
  true_labels = np.argmax(votes, axis=0)
  # accuracy score
  score = np.sum(votes[true_labels, range(n_codes)]) / np.sum(votes)
  if return_model:
    return score, true_labels
  return score
