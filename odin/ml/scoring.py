from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.linalg import eigh, cholesky, inv, svd, solve
import tensorflow as tf

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base import BaseEstimator, TransformerMixin

# ===========================================================================
# Cosine Scoring
# ===========================================================================
def _unit_len_norm(x):
  x_norm = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
  x_norm[x_norm == 0] = 1.
  return x / x_norm

def _wccn(X, y, classes):
  """
  X: [nb_samples, feat_dim]
  y: [nb_samples]
  classes: [0, 1, 2, ...]
  """
  class_avg = np.concatenate([np.mean(X[y == i], axis=0, keepdims=True)
                              for i in classes],
                             axis=0)
  X_mu = X - class_avg[y]
  Sw = np.cov(X_mu.T)
  Sw = Sw + 1e-6 * np.eye(Sw.shape[0])
  w = cholesky(inv(Sw), lower=True)
  return w


class CosineScorer(BaseEstimator, TransformerMixin):
  """ CosineScorer """

  def __init__(self, wccn=True, lda=True):
    super(CosineScorer, self).__init__()
    self._wccn = bool(wccn)
    self._lda = LinearDiscriminantAnalysis() if bool(lda) else None
    self._feat_dim = None
    self._classes = None

  # ==================== properties ==================== #
  @property
  def feat_dim(self):
    return self._feat_dim

  @property
  def classes(self):
    return self._classes

  @property
  def nb_classes(self):
    return len(self._classes)

  @property
  def is_initialized(self):
    return self._feat_dim is not None

  @property
  def is_fitted(self):
    return hasattr(self, '_w')

  @property
  def mean(self):
    return self._mean

  @property
  def w(self):
    return self._w

  @property
  def enroll_vecs(self):
    return self._enroll_vecs

  @property
  def lda(self):
    return self._lda

  # ==================== sklearn ==================== #
  def _initialize(self, X, y):
    if self.is_initialized:
      return
    self._feat_dim = X.shape[1]
    self._classes = np.unique(y)

  def fit(self, X, y):
    if y.ndim == 2:
      y = np.argmax(y, axis=-1)
    self._initialize(X, y)
    # ====== compute classes' average ====== #
    enroll = np.concatenate([np.mean(X[y == i], axis=0, keepdims=True)
                             for i in self.classes], axis=0)
    M = np.mean(X, axis=0, keepdims=True)
    self._mean = M
    X = X - M
    # ====== WCCN ====== #
    if self._wccn:
      w = _wccn(X, y, self.classes) # [feat_dim, feat_dim]
    else:
      w = 1
    self._w = w
    # ====== preprocess ====== #
    # whitening the data
    X = np.dot(X, w)
    # length normalization
    X = _unit_len_norm(X)
    if self._lda is not None:
      self._lda.fit(X, y)
    # ====== enroll vecs ====== #
    enroll = enroll - M
    enroll = np.dot(enroll, w)
    enroll = _unit_len_norm(enroll) # [nb_classes, feat_dim]
    if self._lda is not None:
      enroll = self._lda.transform(enroll) # [nb_classes, nb_classes - 1]
    self._enroll_vecs = _unit_len_norm(enroll)

  def transform(self, X):
    if not self.is_fitted:
      raise RuntimeError("CosineScorer has not been fitted.")
    # ====== preprocess ====== #
    X = X - self._mean
    X = np.dot(X, self._w)
    X = _unit_len_norm(X)
    if self._lda is not None:
      X = self._lda.transform(X)
    # ====== cosine scoring ====== #
    # [nb_classes, nb_classes - 1]
    model_ivectors = self._enroll_vecs
    # [nb_samples, nb_classes - 1]
    test_ivectors = _unit_len_norm(X)
    scores = np.dot(test_ivectors, model_ivectors.T)
    return scores

# ===========================================================================
# PLDA
# ===========================================================================
class PLDA(BaseEstimator, TransformerMixin):
  """ PLDA """

  def __init__(self, arg):
    super(PLDA, self).__init__()
    self.arg = arg
