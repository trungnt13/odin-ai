from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.linalg import eigh, cholesky, inv, svd, solve
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from odin.backend import length_norm, calc_white_mat
from odin.ml.base import BaseEstimator, TransformerMixin, Evaluable

# ===========================================================================
# Cosine Scoring
# ===========================================================================
def compute_class_avg(X, y, classes, sorting=True):
  """ compute average vector for each class

  Parameters
  ----------
  X: [nb_samples, feat_dim]
  y: [nb_samples]
  classes: [nb_classes]
    assumed numerical classes
  sorting: bool
    if True, sort the `classes` by numerical order (from small to large)

  Return
  ------
  [nb_classes, feat_dim]

  Note
  ----
  The given order of each class in `classes` will determine
  the row order of returned matrix
  """
  if sorting:
    classes = sorted(classes, reverse=False)
  return np.concatenate([np.mean(X[y == i], axis=0, keepdims=True)
                         for i in classes],
                        axis=0)

def compute_within_cov(X, y, classes=None, class_avg=None):
  """ Compute the within-classes covariance matrix

  Parameters
  ----------
  X : [nb_samples, feat_dim]
  y : [nb_samples]
  classes : [nb_classes]
    assumed numerical classes
  class_avg : [nb_classes, feat_dim]
    concatenated average vector of each class

  Return
  ------
  [feat_dim, feat_dim]

  Note
  ----
  The given order of each class in `classes` will determine
  the row order of returned matrix
  """
  if classes is None and class_avg is None:
    raise ValueError("`classes` and `class_avg` cannot be None together")
  if classes is not None:
    class_avg = compute_class_avg(X, y, classes, sorting=True)
  X_mu = X - class_avg[y]
  Sw = np.cov(X_mu.T)
  return Sw

def compute_wccn(X, y, classes=None, class_avg=None):
  """ Within class covariance normalization

  Parameters
  ----------
  X : [nb_samples, feat_dim]
  y : [nb_samples]
  classes : [nb_classes]
    assumed numerical classes
  class_avg : [nb_classes, feat_dim]
    concatenated average vector of each class

  Return
  ------
  w: [feat_dim, feat_dim]
    where X_norm = dot(X, w)
  """
  if classes is None and class_avg is None:
    raise ValueError("`classes` and `class_avg` cannot be None together")
  Sw = compute_within_cov(X, y, classes, class_avg)
  Sw = Sw + 1e-6 * np.eye(Sw.shape[0])
  return calc_white_mat(Sw)

class VectorNormalizer(BaseEstimator, TransformerMixin):
  """ Perform of sequence of normalization as following
    -> Centering: Substract sample mean
    -> Whitening: using within-class-covariance-normalization
    -> Applying LDA (optional)
    -> Length normalization

  Parameters
  ----------
  centering : bool (default: True)
    mean normalized the vectors
  wccn : bool (default: True)
    within class covariance normalization
  lda : bool (default: True)
    Linear Discriminant Analysis
  concat : bool (default: False)
    concatenate original vector to the normalized vector

  Return
  ------
  [nb_samples, feat_dim] if `lda=False`
  [nb_samples, nb_classes - 1] if `lda=True` and `concat=False`
  [nb_samples, feat_dim + nb_classes - 1] if `lda=True` and `concat=True`

  """

  def __init__(self, centering=True, wccn=False, unit_length=True,
               lda=False, concat=False):
    super(VectorNormalizer, self).__init__()
    self._centering = bool(centering)
    self._unit_length = bool(unit_length)
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
    return hasattr(self, '_W')

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
  def W(self):
    return self._W

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

  def normalize(self, X, concat=None):
    """
    Parameters
    ----------
    X : array [nb_samples, feat_dim]
    concat : {None, True, False}
      if not None, override the default `concat` attribute of
      this `VectorNormalizer`
    """
    if not self.is_fitted:
      raise RuntimeError("VectorNormalizer has not been fitted.")
    if concat is None:
      concat = self._concat
    if concat:
      X_org = X[:] if not isinstance(X, np.ndarray) else X
    else:
      X_org = None
    # ====== normalizing ====== #
    if self._centering:
      X = X - self._mean
    if self._wccn:
      X = np.dot(X, self.W)
    # ====== LDA ====== #
    if self._lda is not None:
      X_lda = self._lda.transform(X) # [nb_classes, nb_classes - 1]
      # concat if necessary
      if concat:
        X = np.concatenate((X_lda, X_org), axis=-1)
      else:
        X = X_lda
    # ====== unit length normalization ====== #
    if self._unit_length:
      X = length_norm(X, axis=-1, ord=2)
    return X

  def fit(self, X, y):
    y, classes = self._initialize(X, y)
    # ====== compute classes' average ====== #
    enroll = compute_class_avg(X, y, classes, sorting=True)
    M = X.mean(axis=0).reshape(1, -1)
    self._mean = M
    if self._centering:
      X = X - M
    # ====== WCCN ====== #
    if self._wccn:
      W = compute_wccn(X, y, classes=None, class_avg=enroll) # [feat_dim, feat_dim]
    else:
      W = 1
    self._W = W
    # ====== preprocess ====== #
    # whitening the data
    if self._wccn:
      X = np.dot(X, W)
    # length normalization
    if self._unit_length:
      X = length_norm(X, axis=-1)
    # linear discriminant analysis
    if self._lda is not None:
      self._lda.fit(X, y) # [nb_classes, nb_classes - 1]
    # ====== enroll vecs ====== #
    self._enroll_vecs = self.normalize(enroll, concat=False)
    # ====== max min ====== #
    if self._lda is not None:
      X = self._lda.transform(X)
      X = length_norm(X, axis=-1, ord=2)
    vmin = X.min(0, keepdims=True)
    vmax = X.max(0, keepdims=True)
    self._vmin, self._vmax = vmin, vmax
    return self

  def transform(self, X):
    return self.normalize(X)

class Scorer(BaseEstimator, TransformerMixin, Evaluable):
  """ Scorer

  Parameters
  ----------
  centering : bool (default: True)
    mean normalized the vectors
  wccn : bool (default: True)
    within class covariance normalization
  lda : bool (default: True)
    Linear Discriminant Analysis
  concat : bool (default: False)
    concatenate original vector to the normalized vector
  method : {'cosine', 'svm'}
    method for scoring

  """

  def __init__(self, centering=True, wccn=True, lda=True, concat=False,
               method='cosine', labels=None):
    super(Scorer, self).__init__()
    self._normalizer = VectorNormalizer(
        centering=centering, wccn=wccn, lda=lda, concat=concat)
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
    # ====== preprocessing ====== #
    if isinstance(X, (tuple, list)):
      X = np.asarray(X)
    if isinstance(y, (tuple, list)):
      y = np.asarray(y)
    # ====== vector normalizer ====== #
    self._normalizer.fit(X, y)
    if self._labels is None:
      if y.ndim >= 2:
        y = np.argmax(y, axis=-1)
      self._labels = np.unique(y)
    # ====== for SVM method ====== #
    if self.method == 'svm':
      X = self._normalizer.transform(X)
      # normalize to [0, 1]
      X = 2 * (X - self._normalizer.vmin) /\
          (self._normalizer.vmax - self._normalizer.vmin) - 1
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
    return self._svm.predict_proba(self._normalizer.transform(X))

  def predict_log_proba(self, X):
    return self.transform(X)

  def transform(self, X):
    # [nb_samples, nb_classes - 1] (if LDA applied)
    X = self._normalizer.transform(X)
    # ====== cosine scoring ====== #
    if self.method == 'cosine':
      # [nb_classes, nb_classes - 1]
      model_ivectors = self._normalizer.enroll_vecs
      test_ivectors = X
      scores = np.dot(test_ivectors, model_ivectors.T)
    # ====== svm ====== #
    elif self.method == 'svm':
      X = 2 * (X - self._normalizer.vmin) /\
          (self._normalizer.vmax - self._normalizer.vmin) - 1
      scores = self._svm.predict_log_proba(X)
    return scores
