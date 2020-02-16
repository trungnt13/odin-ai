from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import entropy


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
  num_codes = codes.shape[1]
  num_factors = factors.shape[1]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = mutual_info_score(factors[:, j], codes[:, i])
  return m


def discrete_entropy(labels):
  r""" Compute discrete entropy for 1-D or 2-D array.

  If `labels` 2-D array, entropy is calculated for each column

  Arguments:
    labels : 1-D or 2-D array

  Returns:
    entropy : A Scalar or array `[n_factors]`
  """
  labels = np.atleast_1d(labels)
  if labels.ndim == 1:
    return entropy(labels.ravel())
  elif labels.ndim > 2:
    raise ValueError("Only support 1-D or 2-D array for labels entropy.")
  num_factors = labels.shape[1]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = entropy(labels[:, j])
  return h


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
  entropy = discrete_entropy(factors)
  return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))


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
  from sklearn.svm import LinearSVC
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
                               class_weight="balanced",
                               random_state=random_state)
        classifier.fit(x_i[:, np.newaxis], y_j)
        pred = classifier.predict(x_i_test[:, np.newaxis])
        score_matrix[i, j] = np.mean(pred == y_j_test)
  # ====== compute_avg_diff_top_two ====== #
  sorted_matrix = np.sort(score_matrix, axis=0)
  return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


# ===========================================================================
# Disentanglement, completeness, informativeness
# ===========================================================================
def disentanglement_score(importance_matrix):
  r""" Compute the disentanglement score of the representation.

  importance_matrix is of shape [num_codes, num_factors].
  """
  per_code = 1. - sp.stats.entropy(importance_matrix.T + 1e-11,
                                   base=importance_matrix.shape[1])
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
  return np.sum(per_code * code_importance)


def completeness_score(importance_matrix):
  r""""Compute completeness of the representation.

  importance_matrix is of shape [num_codes, num_factors].
  """
  per_factor = 1. - sp.stats.entropy(importance_matrix + 1e-11,
                                     base=importance_matrix.shape[0])
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor * factor_importance)


def dci(repr_train, factor_train, repr_test, factor_test, random_state=1234):
  r""" Disentanglement, completeness, informativeness

  Arguments:
    pass

  Return:
    tuple of 3 scores (disentanglement, completeness, informativeness), all
      scores are higher is better.

  Note:
    This impelentation only return accuracy on test data as informativeness
      score
  """
  num_factors = factor_train.shape[1]
  num_codes = repr_train.shape[1]
  # ====== compute importance based on gradient boosted trees ====== #
  importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
  train_acc = []
  test_acc = []
  for i in range(num_factors):
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(repr_train, factor_train[:, i])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_acc.append(np.mean(model.predict(repr_train) == factor_train[:, i]))
    test_acc.append(np.mean(model.predict(repr_test) == factor_test[:, i]))
  train_acc = np.mean(train_acc)
  test_acc = np.mean(test_acc)
  # ====== disentanglement and completeness ====== #
  d = disentanglement_score(importance_matrix)
  c = completeness_score(importance_matrix)
