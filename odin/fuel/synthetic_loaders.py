# This script generate synthetic data
from __future__ import print_function, division, absolute_import

import numpy as np

from odin.fuel.dataset import Dataset

# ===========================================================================
# Helpers
# ===========================================================================
def rand_probabilities(p_min=0. + 10e-8, p_max=1. - 10e-8,
                       shape=(), seed=None):
  if isinstance(seed, np.random.RandomState):
    rand = seed
  else:
    rand = np.random.RandomState(seed=seed)
  return rand.rand(*shape) * (p_max - p_min) + p_min

def rand_samples_per_distribution(n_total, n_dist, seed=None):
  if isinstance(seed, np.random.RandomState):
    rand = seed
  else:
    rand = np.random.RandomState(seed=seed)
  if n_dist == 1:
    n_samples_per_distribution = (n_total,)
  else: # stick breaking
    n_samples_per_distribution = sorted(rand.choice(
        a=np.arange(n_total), size=n_dist - 1, replace=False)) + [n_total]
    n_samples_per_distribution = (n_samples_per_distribution[0],) + \
    tuple([i1 - i0
        for i0, i1, in zip(n_samples_per_distribution, n_samples_per_distribution[1:])])
  assert np.sum(n_samples_per_distribution) == n_total
  return n_samples_per_distribution

# ===========================================================================
# Synthesizing datasets
# ===========================================================================
def generate_ZeroInflatedNegativeBinomial(
    n_samples=12000, n_features=8, n_labels=3,
    n=[5, 25], p_success=[0.1, 0.5], p_drop=[0.2, 0.5],
    seed=52181208):
  rand = np.random.RandomState(seed)
  n_samples_per_distribution = rand_samples_per_distribution(
      n_total=n_samples + 1000, n_dist=n_labels)
  # ====== sampling ====== #
  X = []
  y = []
  params = []
  for i in range(n_labels):
    # random parameters
    shape = (n_samples_per_distribution[i], n_features)
    total_count = rand.randint(n[0], n[1] + 1)
    prob_sucesss = rand_probabilities(p_min=p_success[0],
                                      p_max=p_success[1],
                                      seed=rand)
    prob_keep = rand_probabilities(p_min=1. - p_drop[1],
                                   p_max=1. - p_drop[0],
                                   seed=rand)
    params.append((total_count, prob_sucesss, 1 - prob_keep))
    # random data
    data = rand.negative_binomial(n=total_count, p=prob_sucesss, size=shape)
    mask = np.random.binomial(n=1, p=prob_keep, size=shape)
    newdata = (data * mask)
    indices_to_keep = (newdata.sum(axis=1) > 0).ravel()
    newdata = newdata[indices_to_keep]
    X.append(newdata)
    y += [i] * shape[0]
  # ====== post processing ====== #
  X = np.concatenate(X, axis=0)
  y = np.array(y)
  ids = rand.permutation(X.shape[0])[:n_samples]
  X = X[ids]
  y = y[ids]
  assert X.shape[0] == y.shape[0] == n_samples
  return X, y, params

def generate_ZeroInflatedPoisson(
    n_samples=12000, n_features=8, n_labels=3,
    mu=[5, 25], theta=[2, 8], p_drop=[0.2, 0.5],
    seed=52181208):
  rand = np.random.RandomState(seed)
  n_samples_per_distribution = rand_samples_per_distribution(
      n_total=n_samples + 1000, n_dist=n_labels)
  # ====== sampling ====== #
  # mean Gamma: shape * scale
  X = []
  y = []
  params = []
  for i in range(n_labels):
    # random parameters
    shape = (n_samples_per_distribution[i], n_features)
    mu_ = rand.randint(mu[0], mu[1] + 1)
    theta_ = rand.randint(theta[0], theta[1] + 1)
    p = mu_ / (mu_ + theta_)
    r = theta_
    prob_keep = rand_probabilities(p_min=1. - p_drop[1],
                                   p_max=1. - p_drop[0],
                                   seed=rand)
    params.append((mu_, theta_, 1 - prob_keep))
    # random data
    data = np.random.poisson(
        np.random.gamma(r, p / (1 - p), size=shape)
    )
    mask = np.random.binomial(n=1, p=prob_keep, size=shape)
    newdata = (data * mask)
    indices_to_keep = (newdata.sum(axis=1) > 0).ravel()
    newdata = newdata[indices_to_keep]
    X.append(newdata)
    y += [i] * shape[0]
  # ====== post processing ====== #
  X = np.concatenate(X, axis=0)
  y = np.array(y)
  ids = rand.permutation(X.shape[0])[:n_samples]
  X = X[ids]
  y = y[ids]
  assert X.shape[0] == y.shape[0] == n_samples
  return X, y, params
