from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.linalg import eigh, cholesky, inv, svd, solve
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base import BaseEstimator, TransformerMixin, Evaluable

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

class VectorNormalization(BaseEstimator, TransformerMixin):
  """

  Parameters
  ----------
  wccn : bool (default: True)
    within class covariance normalization
  lda : bool (default: True)
    Linear Discriminant Analysis
  concat : bool (default: False)
    concatenate original vector to the transformed

  Return
  ------
  [nb_samples, feat_dim] if `lda=False`
  [nb_samples, nb_classes - 1] if `lda=True` and `concat=False`
  [nb_samples, feat_dim + nb_classes - 1] if `lda=True` and `concat=True`

  """

  def __init__(self, wccn=True, lda=True, concat=False):
    super(VectorNormalization, self).__init__()
    self._wccn = bool(wccn)
    self._lda = LinearDiscriminantAnalysis() if bool(lda) else None
    self._feat_dim = None
    self._concat = bool(concat)

  # ==================== properties ==================== #
  @property
  def feat_dim(self):
    return self._feat_dim

  @property
  def is_initialized(self):
    return self._feat_dim is not None

  @property
  def is_fitted(self):
    return hasattr(self, '_w')

  @property
  def enroll_vecs(self):
    return self._enroll_vecs

  @property
  def mean(self):
    """ global mean vector """
    return self._mean

  @property
  def vmin(self):
    return self._vmin

  @property
  def vmax(self):
    return self._vmax

  @property
  def w(self):
    return self._w

  @property
  def lda(self):
    return self._lda

  # ==================== sklearn ==================== #
  def _initialize(self, X, y):
    if not self.is_initialized:
      self._feat_dim = X.shape[1]
    assert self._feat_dim == X.shape[1]
    if isinstance(y, (tuple, list)):
      y = np.asarray(y)
    if y.ndim == 2:
      y = np.argmax(y, axis=-1)
    return y, np.unique(y)

  def normalize(self, X):
    if not self.is_fitted:
      raise RuntimeError("VectorNormalization has not been fitted.")
    X_org = X[:] if not isinstance(X, np.ndarray) else X
    X = X - self._mean
    X = np.dot(X, self._w)
    X = _unit_len_norm(X)
    # ====== LDA ====== #
    if self._lda is not None:
      X_lda = self._lda.transform(X)
      # concat if necessary
      if self._concat:
        X = np.concatenate((X_lda, X_org), axis=-1)
      else:
        X = X_lda
    return X

  def fit(self, X, y):
    y, y_uni = self._initialize(X, y)
    # ====== compute classes' average ====== #
    enroll = np.concatenate([np.mean(X[y == i], axis=0, keepdims=True)
                             for i in y_uni], axis=0)
    M = X.mean(axis=0).reshape(1, -1)
    self._mean = M
    X = X - M
    # ====== WCCN ====== #
    if self._wccn:
      w = _wccn(X, y, y_uni) # [feat_dim, feat_dim]
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
    # ====== max min ====== #
    if self._lda is not None:
      X = self._lda.transform(X)
    vmin = X.min(0, keepdims=True)
    vmax = X.max(0, keepdims=True)
    self._vmin, self._vmax = vmin, vmax
    return self

  def transform(self, X):
    return self.normalize(X)

class Scorer(BaseEstimator, TransformerMixin, Evaluable):
  """ Scorer """

  def __init__(self, wccn=True, lda=True, concat=False,
               method='cosine', labels=None):
    super(Scorer, self).__init__()
    self._normalizer = VectorNormalization(wccn=wccn, lda=lda,
                                           concat=concat)
    self._labels = labels
    method = str(method).lower()
    if method not in ('cosine', 'svm'):
      raise ValueError('`method` must be one of the following: cosine, svm; '
                       'but given: "%s"' % method)
    self._method = method

  # ==================== properties ==================== #
  @property
  def method(self):
    return self._method

  @property
  def feat_dim(self):
    return self._normalizer.feat_dim

  @property
  def labels(self):
    return self._labels

  @property
  def nb_classes(self):
    return len(self._labels)

  @property
  def is_initialized(self):
    return self._normalizer.is_initialized

  @property
  def is_fitted(self):
    return self._normalizer.is_fitted

  @property
  def normalizer(self):
    return self._normalizer

  @property
  def lda(self):
    return self._normalizer.lda

  # ==================== sklearn ==================== #
  def fit(self, X, y):
    self._normalizer.fit(X, y)
    if self._labels is None:
      if y.ndim >= 2:
        y = np.argmax(y, axis=-1)
      self._labels = np.unique(y)
    # ====== for SVM method ====== #
    if self.method == 'svm':
      X = X - self.normalizer.mean
      # whitening the data
      X = np.dot(X, self.normalizer.w)
      # length normalization
      X = _unit_len_norm(X)
      if self.lda is not None:
        X = self.lda.transform(X)
      # normalize to [0, 1]
      X = 2 * (X - self.normalizer.vmin) /\
          (self.normalizer.vmax - self.normalizer.vmin) - 1
      self._svm = SVC(C=1, kernel='rbf', gamma='auto', coef0=1,
                      shrinking=True, random_state=0,
                      probability=True, tol=1e-3,
                      cache_size=1e4, class_weight='balanced')
      self._svm.fit(X, y)
      self.predict_proba = self._predict_proba
    return self

  def _predict_proba(self, X):
    if self.method != 'svm':
      raise RuntimeError("`predict_proba` only for 'svm' method")
    return self._svm.predict_proba(self.normalizer.transform(X))

  def predict_log_proba(self, X):
    return self.transform(X)

  def transform(self, X):
    X = self.normalizer.transform(X)
    # ====== cosine scoring ====== #
    if self.method == 'cosine':
      # [nb_classes, nb_classes - 1]
      model_ivectors = self.normalizer.enroll_vecs
      # [nb_samples, nb_classes - 1]
      test_ivectors = _unit_len_norm(X)
      scores = np.dot(test_ivectors, model_ivectors.T)
    # ====== svm ====== #
    elif self.method == 'svm':
      X = 2 * (X - self.normalizer.vmin) /\
          (self.normalizer.vmax - self.normalizer.vmin) - 1
      scores = self._svm.predict_log_proba(X)
    return scores

# ===========================================================================
# PLDA
# ===========================================================================
class PLDA(BaseEstimator, TransformerMixin):
  """ PLDA """

  def __init__(self, arg):
    super(PLDA, self).__init__()
    self.arg = arg
