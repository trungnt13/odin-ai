# -*- coding: utf-8 -*-
""""
author: 'Omid Sadjadi, Timothee Kheyrkhah'
email: 'omid.sadjadi@nist.gov'
"""
import time
import warnings
from numbers import Number
from six import string_types

import numpy as np
from scipy.linalg import eigh, cholesky, inv, svd, solve

from odin.backend import length_norm, calc_white_mat
from odin.utils import unique
from odin.ml.base import BaseEstimator, TransformerMixin, Evaluable
from odin.ml.scoring import (compute_within_cov, compute_class_avg,
                             VectorNormalizer)

def logdet(A):
  u = cholesky(A)
  y = 2 * np.log(np.diag(u)).sum()
  return y

class PLDA(BaseEstimator, TransformerMixin, Evaluable):
  """ Probabilistic LDA

  Parameters
  ----------
  n_phi : int
    number of dimension for the latent space

  centering : bool (default: True)
    mean normalization the data before EM

  wccn : bool (default: True)
    within class covariance normalization before EM

  unit_length : bool (default: True)
    normalize vector length of each sample to 1 before EM

  n_iter : {integer, 'auto'}
    if 'auto', keep iterating until no more improvement (i.e. reduction in `sigma` value)
    compared to the `improve_threshold`

  improve_threshold : scalar
    Only used in case `n_iter='auto'`

  labels : {list of string, or None} (default: None)
    labels information for `evaluate` method

  seed : int
    random seed for reproducibility

  verbose : int (default: 0)
    verbose level, 0 for turning off all logging activities,
    1 for basics notification, 2 for fitting progress.
    if `2`, compute log-likelihood during fitting EM, this will
    significantly slows down the process, only suggested for debugging

  Attributes
  ----------
  Sigma_ : [feat_dim, feat_dim]
  Phi_ : [feat_dim, n_phi]
  Sb_ : [feat_dim, feat_dim]
  St_ : [feat_dim, feat_dim]
  Lambda : []
  Uk : []
  Q_hat : []
  X_model_ : [num_class, feat_dim]
    class-dependence feature vectors
  """

  def __init__(self, n_phi=None,
               centering=True, wccn=True, unit_length=True,
               n_iter='auto', improve_threshold=1e-1,
               labels=None, dtype='float64', random_state=None,
               verbose=0):
    super(PLDA, self).__init__()
    # ====== check n_phi ====== #
    if n_phi is not None:
      n_phi = int(n_phi)
    self.n_phi_ = n_phi
    # ====== check num_iter ====== #
    if isinstance(n_iter, string_types):
      n_iter = n_iter.lower()
      assert n_iter == 'auto', 'Invalid `n_iter` value: %s' % n_iter
    elif isinstance(n_iter, Number):
      assert n_iter > 0, "`n_iter` must greater than 0, but given: %d" % n_iter
    self.n_iter_ = n_iter
    self.improve_threshold_ = float(improve_threshold)
    # ====== other ====== #
    self.feat_dim_ = None
    self._labels = labels
    self.verbose_ = int(verbose)
    # for normalization
    self._normalizer = VectorNormalizer(
        centering=centering, wccn=wccn, unit_length=unit_length,
        lda=False, concat=False)
    self._dtype = np.dtype(dtype)
    # ====== check random state ====== #
    if random_state is None:
      self._rand_state = np.random.RandomState(None)
    elif isinstance(random_state, Number):
      self._rand_state = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
      self._rand_state = random_state
    else:
      raise ValueError("Invalid argument for `random_state`: %s" % str(random_state))
    # Attributes
    self.Sigma_ = None
    self.Phi_ = None
    self.Sb_ = None
    self.St_ = None

  # ==================== properties ==================== #
  @property
  def dtype(self):
    return self._dtype

  @property
  def feat_dim(self):
    return self.feat_dim_

  @property
  def normalizer(self):
    return self._normalizer

  @property
  def labels(self):
    return self._labels

  @property
  def num_classes(self):
    return len(self._labels)

  @property
  def is_fitted(self):
    if not hasattr(self, 'Lambda_') or \
    not hasattr(self, 'Uk_') or \
    not hasattr(self, 'Q_hat_') or \
    not hasattr(self, 'X_model_'):
      return False
    return True

  # ==================== Pickling ==================== #
  def __getstate__(self):
    if not self.is_fitted:
      raise RuntimeError("The PLDA have not been fitted, nothing to pickle!")
    return (self.n_phi_, self.n_iter_, self.feat_dim_, self._labels, self.verbose_,
            self._normalizer, self._dtype, self._rand_state,
            self.Sigma_, self.Phi_, self.Sb_, self.St_,
            self.Lambda_, self.Uk_, self.Q_hat_, self.X_model_)

  def __setstate__(self, states):
    (self.n_phi_, self.n_iter_, self.feat_dim_, self._labels, self.verbose_,
     self._normalizer, self._dtype, self._rand_state,
     self.Sigma_, self.Phi_, self.Sb_, self.St_,
     self.Lambda_, self.Uk_, self.Q_hat_, self.X_model_) = states

  # ==================== helpers ==================== #
  def initialize(self, X, labels):
    feat_dim = X.shape[1]
    if self.feat_dim is None or self._num_classes is None:
      self.feat_dim_ = int(feat_dim)
      if self._labels is None:
        self._labels = labels
      if self.feat_dim <= self.n_phi_:
        raise RuntimeError("`feat_dim=%d` must be greater than `n_phi=%d`" %
          (self.feat_dim, self.n_phi_))
      # ====== initialize ====== #
      # covariance matrix of the residual term
      # self.Sigma_ = 1. / self.feat_dim * np.eye(self.feat_dim, dtype=self.dtype)
      self.Sigma_ = (1. / self.feat_dim * np.eye(self.feat_dim) +
                    self._rand_state.randn(self.feat_dim, self.feat_dim)
                    ).astype(self.dtype)
      # self.Sigma_ = np.cov(X.T).astype(self.dtype)
      # self.Sigma_ = (np.cov(X.T) +
      #               self._rand_state.randn(self.feat_dim, self.feat_dim)
      #               ).astype(self.dtype)
      # self.Sigma_ = 100 * self._rand_state.randn(
      #     self.feat_dim, self.feat_dim).astype(self.dtype)
      # factor loading matrix (Eignevoice matrix) [feat_dim, n_phi]
      # self.Phi_ = np.r_[np.eye(self.n_phi_),
      #                  np.zeros((self.feat_dim - self.n_phi_, self.n_phi_))]
      # self.Phi_ = self._rand_state.randn(self.feat_dim, self.n_phi_).astype(self.dtype)
      self.Phi_ = self.normalizer.transform(
          self._rand_state.randn(self.n_phi_, self.feat_dim)
      ).T.astype(self.dtype)
      self.Sb_ = np.zeros((self.feat_dim, self.feat_dim), dtype=self.dtype)
      self.St_ = np.zeros((self.feat_dim, self.feat_dim), dtype=self.dtype)
    # ====== validate the dimension ====== #
    if self.feat_dim != feat_dim:
      raise ValueError("Mismatch the input feature dimension, %d != %d" %
        (self.feat_dim, feat_dim))
    if self.num_classes != len(labels):
      raise ValueError("Mismatch the number of output classes, %d != %d" %
        (self.num_classes, len(labels)))

  # ==================== sklearn ==================== #
  def _update_caches(self):
    # ====== update cached matrices for scoring ====== #
    iSt = inv(self.St_) # [feat_dim, feat_dim]
    iS = inv(self.St_ - np.dot(np.dot(self.Sb_, iSt), self.Sb_))
    Q = iSt - iS # [feat_dim, feat_dim]
    P = np.dot(np.dot(iSt, self.Sb_), iS) # [feat_dim, feat_dim]
    U, s, V = svd(P, full_matrices=False)
    self.Lambda_ = np.diag(s[:self.n_phi_]) # [n_phi, n_phi]
    self.Uk_ = U[:, :self.n_phi_] # [feat_dim, n_phi]
    self.Q_hat_ = np.dot(np.dot(self.Uk_.T, Q), self.Uk_) # [n_phi, n_phi]

  def fit_maximum_likelihood(self, X, y):
    # ====== preprocessing ====== #
    if isinstance(X, (tuple, list)):
      X = np.asarray(X)
    elif "odin.fuel" in str(type(X)):
      X = X[:]
    if isinstance(y, (tuple, list)):
      y = np.asarray(y)
    # ====== normalizing and initializing ====== #
    X = self.normalizer.fit(X, y).transform(X)
    classes = np.unique(y)
    self.initialize(X, labels=classes)
    # ====== ml ====== #
    Sw = compute_within_cov(X, y, classes)
    self.St_ = np.cov(X.T)
    self.Sb_ = self.St_ - Sw
    # ====== the default class_avg ====== #
    self._update_caches()
    model_vecs = compute_class_avg(X, y, classes=classes)
    self.X_model_ = np.dot(model_vecs, self.Uk_)
    return self

  def fit(self, X, y):
    """
    Parameters
    ----------
    X : [num_samples, feat_dim]
    y : [num_samples]
    """
    # ====== preprocessing ====== #
    if isinstance(X, (tuple, list)):
      X = np.asarray(X)
    elif "odin.fuel" in str(type(X)):
      X = X[:]
    if isinstance(y, (tuple, list)):
      y = np.asarray(y)
    assert X.shape[0] == y.shape[0], \
        "Number of samples mismatch in `X` and `y`, %d != %d" % \
        (X.shape[0], y.shape[0])
    # ====== normalize and initialize ====== #
    y_counts = np.bincount(y) # sessions per speaker
    classes = np.unique(y)
    X = self.normalizer.fit(X, y).transform(X)
    self.initialize(X, labels=classes)
    # ====== Initializing ====== #
    F = np.zeros((self.num_classes, self.feat_dim))
    for clz in np.unique(y):
      # Speaker indices
      F[clz, :] = X[y == clz, :].sum(axis=0)
    if self.verbose_ > 0:
      print('Re-estimating the Eigenvoice subspace with {} factors ...'.format(self.n_phi_))
    X_sqr = np.dot(X.T, X)
    # ====== iteration ====== #
    iter = 0
    last_llk_value = None
    while True:
      e_time = time.time()
      # expectation
      Ey, Eyy = self.expectation_plda(F, y_counts)
      e_time = time.time() - e_time
      # maximization
      m_time = time.time()
      self.maximization_plda(X, X_sqr, F, Ey, Eyy)
      m_time = time.time() - m_time
      # log-likelihood
      llk = 'None'
      llk_value = None
      if self.verbose_ > 1 or isinstance(self.n_iter_, string_types):
        llk_value = self.compute_llk(X)
        llk = '%.2f' % llk_value
      if self.verbose_ > 0:
        print('#iter:%-3d \t [llk = %s] \t [E-step = %.2f s] [M-step = %.2f s]' %
              (iter + 1, llk, e_time, m_time))
      # check breaking condition
      iter += 1
      if isinstance(self.n_iter_, Number):
        if iter >= self.n_iter_:
          break
      elif iter > 2 and last_llk_value is not None:
        if llk_value - last_llk_value < self.improve_threshold_:
          break
      last_llk_value = llk_value
    # ====== Update the eigenvoice space ====== #
    self.Sb_ = self.Phi_.dot(self.Phi_.T)
    self.St_ = self.Sb_ + self.Sigma_
    # ====== the default class_avg ====== #
    self._update_caches()
    model_vecs = compute_class_avg(X, y, classes=classes)
    self.X_model_ = np.dot(model_vecs, self.Uk_)

  def expectation_plda(self, F, cls_counts):
    """
    Parameters
    ----------
    F : [num_classes, feat_dim]
    cls_count : [num_classes]
    """
    # computes the posterior mean and covariance of the factors
    num_classes = F.shape[0]
    Eyy = np.zeros(shape=(self.n_phi_, self.n_phi_))
    Ey_clz = np.zeros(shape=(num_classes, self.n_phi_))
    # initialize common terms to save computations
    uniqFreqs = unique(cls_counts, keep_order=True)
    n_uniq = len(uniqFreqs)
    invTerms = np.empty(shape=(n_uniq, self.n_phi_, self.n_phi_))
    PhiT_invS = solve(self.Sigma_.T, self.Phi_).T # [n_phi, feat_dim]
    PhiT_invS_Phi = np.dot(PhiT_invS, self.Phi_) # [n_phi, n_phi]
    I = np.eye(self.n_phi_)

    for ix in range(n_uniq):
      nPhiT_invS_Phi = uniqFreqs[ix] * PhiT_invS_Phi
      invTerms[ix] = inv(I + nPhiT_invS_Phi)

    for clz in range(num_classes):
      num_samples = cls_counts[clz]
      PhiT_invS_y = np.dot(PhiT_invS, F[clz, :])
      idx = np.flatnonzero(uniqFreqs == num_samples)[0]
      Cyy = invTerms[idx]
      Ey_clz[clz, :] = np.dot(Cyy, PhiT_invS_y)
      Eyy += num_samples * Cyy

    Eyy += np.dot((Ey_clz * cls_counts[:, None]).T, Ey_clz)
    return Ey_clz, Eyy

  def compute_llk(self, X):
    """
    Parameters
    ----------
    X : [num_samples, feat_dim]
    """
    num_samples = X.shape[0]
    S = np.dot(self.Phi_, self.Phi_.T) + self.Sigma_ # [feat_dim, feat_dim]
    llk = -0.5 * (self.feat_dim * num_samples * np.log(2 * np.pi) +
                  num_samples * logdet(S) +
                  np.sum(X * solve(S, X.T).T))
    return llk

  def maximization_plda(self, X, X_sqr, F, Ey, Eyy):
    """
    ML re-estimation of the Eignevoice subspace and the covariance of the
    residual noise (full).

    Paremters
    ---------
    X : [num_samples, feat_dim]
    X_cov : [feat_dim, feat_dim]
    F : [num_classes, feat_dim]
    Ey : [num_classes, n_phi]
    Eyy : [n_phi, n_phi]
    """
    num_samples = X.shape[0]
    Ey_FT = np.dot(Ey.T, F) # [n_phi, feat_dim]
    self.Phi_ = solve(Eyy.T, Ey_FT).T # [feat_dim, n_phi]
    self.Sigma_ = 1. / num_samples * (X_sqr - np.dot(self.Phi_, Ey_FT))

  def transform(self, X):
    if not self.is_fitted:
      raise RuntimeError("This model hasn't been fitted!")
    # ====== check X ====== #
    if isinstance(X, (tuple, list)):
      X = np.asarray(X)
    elif "odin.fuel" in str(type(X)):
      X = X[:]
    # ====== transform into latent space ====== #
    X_norm = self.normalizer.transform(X)
    X_project = np.dot(X_norm, self.Uk_) # [num_samples, n_phi]
    return X_project
    # return np.dot(X_project, self.Q_hat_)
    # h = np.dot(X_project, self.Q_hat_) * X_project
    # return h

  def predict_log_proba(self, X, X_model=None):
    """
    Parameters
    ----------
    X : [num_samples, feat_dim]
    X_model : [num_classes, feat_dim]
      if None, use class average extracted based on fitted data

    Return
    ------
    log-probabilities matrix [num_samples, num_classes]
    """
    if not self.is_fitted:
      raise RuntimeError("This model hasn't been fitted!")
    # ====== check X_model ====== #
    if X_model is None:
      X_model = self.X_model_
    else:
      # [num_classes, n_phi]
      X_model = np.dot(self.normalizer.transform(X_model), self.Uk_)
    if X_model.shape[0] != self.num_classes:
      warnings.warn("The model matrix contains %d classes, but the "
                    "fitted number of classes is %d" %
                    (X_model.shape[0], self.num_classes))
    # ====== check X ====== #
    if isinstance(X, (tuple, list)):
      X = np.asarray(X)
    elif "odin.fuel" in str(type(X)):
      X = X[:]
    # ====== transform the input matrices ====== #
    X = np.dot(self.normalizer.transform(X), self.Uk_) # [num_samples, n_phi]
    # [num_classes, 1]
    score_h1 = np.sum(np.dot(X_model, self.Q_hat_) * X_model, axis=1, keepdims=True)
    # [num_samples, 1]
    score_h2 = np.sum(np.dot(X, self.Q_hat_) * X, axis=1, keepdims=True)
    # [num_samples, num_classes]
    score_h1h2 = 2 * np.dot(X, np.dot(X_model, self.Lambda_).T)
    # [num_samples, num_classes]
    scores = score_h1h2 + score_h1.T + score_h2
    return scores
