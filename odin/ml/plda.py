# -*- coding: utf-8 -*-
""""
author: 'Omid Sadjadi, Timothee Kheyrkhah'
email: 'omid.sadjadi@nist.gov'
"""
import time

import numpy as np
from scipy.linalg import eigh, cholesky, inv, svd, solve

from odin.backend import length_norm, calc_white_mat
from odin.utils import unique, Progbar
from .base import BaseEstimator, TransformerMixin, Evaluable
from .scoring import (compute_within_cov, compute_class_avg,
                      VectorNormalizer)

def logdet(A):
  u = cholesky(A)
  y = 2 * np.log(np.diag(u)).sum()
  return y

class PLDA(BaseEstimator, TransformerMixin, Evaluable):
  """ Probabilistic LDA

  Parameters
  ----------
  num_phi : int
    number of dimension for the latent space
  num_iter : int (default: 12)
    number of EM iteration
  centering : bool (default: True)
    mean normalization the data before EM
  wccn : bool (default: True)
    within class covariance normalization before EM
  unit_length : bool (default: True)
    normalize vector length of each sample to 1 before EM
  labels : {list of string, or None} (default: None)
    labels information for `evaluate` method
  seed : int
    random seed for reproducibility
  show_llk : bool (default: False)
    if True, compute log-likelihood during fitting EM,
    this will significantly slows down the process,
    only suggested for debugging

  Attributes
  ----------
  Sigma : [feat_dim, feat_dim]
  Phi : [feat_dim, num_phi]
  Sb : [feat_dim, feat_dim]
  St : [feat_dim, feat_dim]
  Lambda : []
  Uk : []
  Q_hat : []
  X_model : []
  """

  def __init__(self, num_phi, num_iter=12,
               centering=True, wccn=True, unit_length=True,
               labels=None, dtype='float64', seed=5218,
               show_llk=False):
    super(PLDA, self).__init__()
    self.num_phi = int(num_phi)
    self.num_iter = int(num_iter)
    self._feat_dim = None
    self._labels = labels
    self.show_llk = bool(show_llk)
    # for normalization
    self._normalizer = VectorNormalizer(
        centering=centering, wccn=wccn, unit_length=unit_length,
        lda=False, concat=False)
    self._dtype = np.dtype(dtype)
    self._rand_state = np.random.RandomState(seed=seed)
    # Attributes
    self.Sigma = None
    self.Phi = None
    self.Sb = None
    self.St = None

  # ==================== properties ==================== #
  @property
  def dtype(self):
    return self._dtype

  @property
  def feat_dim(self):
    return self._feat_dim

  @property
  def normalizer(self):
    return self._normalizer

  @property
  def labels(self):
    return self._labels

  @property
  def nb_classes(self):
    return len(self._labels)

  @property
  def is_fitted(self):
    if not hasattr(self, 'Lambda') or \
    not hasattr(self, 'Uk') or \
    not hasattr(self, 'Q_hat') or \
    not hasattr(self, 'X_model'):
      return False
    return True

  # ==================== Pickling ==================== #
  def __getstate__(self):
    if not self.is_fitted:
      raise RuntimeError("The PLDA have not been fitted, nothing to pickle!")
    return (self.num_phi, self.num_iter, self._feat_dim, self._labels, self.show_llk,
            self._normalizer, self._dtype, self._rand_state,
            self.Sigma, self.Phi, self.Sb, self.St,
            self.Lambda, self.Uk, self.Q_hat, self.X_model)

  def __setstate__(self, states):
    (self.num_phi, self.num_iter, self._feat_dim, self._labels, self.show_llk,
     self._normalizer, self._dtype, self._rand_state,
     self.Sigma, self.Phi, self.Sb, self.St,
     self.Lambda, self.Uk, self.Q_hat, self.X_model) = states

  # ==================== helpers ==================== #
  def initialize(self, X, labels):
    feat_dim = X.shape[1]
    if self.feat_dim is None or self._nb_classes is None:
      self._feat_dim = int(feat_dim)
      if self._labels is None:
        self._labels = labels
      if self.feat_dim <= self.num_phi:
        raise RuntimeError("`feat_dim=%d` must be greater than `num_phi=%d`" %
          (self.feat_dim, self.num_phi))
      # ====== initialize ====== #
      # covariance matrix of the residual term
      # self.Sigma = 1. / self.feat_dim * np.eye(self.feat_dim, dtype=self.dtype)
      self.Sigma = (1. / self.feat_dim * np.eye(self.feat_dim) +
                    self._rand_state.randn(self.feat_dim, self.feat_dim)
                    ).astype(self.dtype)
      # self.Sigma = np.cov(X.T).astype(self.dtype)
      # self.Sigma = (np.cov(X.T) +
      #               self._rand_state.randn(self.feat_dim, self.feat_dim)
      #               ).astype(self.dtype)
      # self.Sigma = 100 * self._rand_state.randn(
      #     self.feat_dim, self.feat_dim).astype(self.dtype)
      # factor loading matrix (Eignevoice matrix) [feat_dim, num_phi]
      # self.Phi = np.r_[np.eye(self.num_phi),
      #                  np.zeros((self.feat_dim - self.num_phi, self.num_phi))]
      # self.Phi = self._rand_state.randn(self.feat_dim, self.num_phi).astype(self.dtype)
      self.Phi = self.normalizer.transform(
          self._rand_state.randn(self.num_phi, self.feat_dim)
      ).T.astype(self.dtype)
      self.Sb = np.zeros((self.feat_dim, self.feat_dim), dtype=self.dtype)
      self.St = np.zeros((self.feat_dim, self.feat_dim), dtype=self.dtype)
    # ====== validate the dimension ====== #
    if self.feat_dim != feat_dim:
      raise ValueError("Mismatch the input feature dimension, %d != %d" %
        (self.feat_dim, feat_dim))
    if self.nb_classes != len(labels):
      raise ValueError("Mismatch the number of output classes, %d != %d" %
        (self.nb_classes, len(labels)))

  # ==================== sklearn ==================== #
  def _update_caches(self):
    # ====== update cached matrices for scoring ====== #
    iSt = inv(self.St) # [feat_dim, feat_dim]
    iS = inv(self.St - np.dot(np.dot(self.Sb, iSt), self.Sb))
    Q = iSt - iS # [feat_dim, feat_dim]
    P = np.dot(np.dot(iSt, self.Sb), iS) # [feat_dim, feat_dim]
    U, s, V = svd(P, full_matrices=False)
    self.Lambda = np.diag(s[:self.num_phi]) # [num_phi, num_phi]
    self.Uk = U[:, :self.num_phi] # [feat_dim, num_phi]
    self.Q_hat = np.dot(np.dot(self.Uk.T, Q), self.Uk) # [num_phi, num_phi]

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
    self.St = np.cov(X.T)
    self.Sb = self.St - Sw
    # ====== the default class_avg ====== #
    self._update_caches()
    model_vecs = compute_class_avg(X, y, classes=classes)
    self.X_model = np.dot(model_vecs, self.Uk)
    return self

  def fit(self, X, y):
    """
    Parameters
    ----------
    X : [nb_samples, feat_dim]
    y : [nb_samples]
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
    F = np.zeros((self.nb_classes, self.feat_dim))
    for clz in np.unique(y):
      # Speaker indices
      F[clz, :] = X[y == clz, :].sum(axis=0)
    print('Re-estimating the Eigenvoice subspace with {} factors ...\n'.format(self.num_phi))
    prog = Progbar(target=self.num_iter, print_report=True, print_summary=True,
                   name='PLDA-EM')
    X_sqr = np.dot(X.T, X)
    for iter in range(self.num_iter):
      e_time = time.time()
      # expectation
      Ey, Eyy = self.expectation_plda(F, y_counts)
      e_time = time.time() - e_time
      # maximization
      m_time = time.time()
      self.maximization_plda(X, X_sqr, F, Ey, Eyy)
      m_time = time.time() - m_time
      # log-likelihood
      if self.show_llk:
        llk = self.compute_llk(X)
        print('EM iter#: {} \t'.format(iter + 1), end=" ")
        print('[llk = {0:.2f}] \t [E-step = {1:.2f} s] [M-step = {1:.2f} s]'.format(
              llk, e_time, m_time))
      else:
        prog['E-step'] = e_time
        prog['M-step'] = m_time
        prog.add(1)
    # ====== Update the eigenvoice space ====== #
    self.Sb = self.Phi.dot(self.Phi.T)
    self.St = self.Sb + self.Sigma
    # ====== the default class_avg ====== #
    self._update_caches()
    model_vecs = compute_class_avg(X, y, classes=classes)
    self.X_model = np.dot(model_vecs, self.Uk)

  def expectation_plda(self, F, cls_counts):
    """
    Parameters
    ----------
    F : [nb_classes, feat_dim]
    cls_count : [nb_classes]
    """
    # computes the posterior mean and covariance of the factors
    nb_classes = F.shape[0]
    Eyy = np.zeros(shape=(self.num_phi, self.num_phi))
    Ey_clz = np.zeros(shape=(nb_classes, self.num_phi))
    # initialize common terms to save computations
    uniqFreqs = unique(cls_counts, keep_order=True)
    n_uniq = len(uniqFreqs)
    invTerms = np.empty(shape=(n_uniq, self.num_phi, self.num_phi))
    PhiT_invS = solve(self.Sigma.T, self.Phi).T # [num_phi, feat_dim]
    PhiT_invS_Phi = np.dot(PhiT_invS, self.Phi) # [num_phi, num_phi]
    I = np.eye(self.num_phi)

    for ix in range(n_uniq):
      nPhiT_invS_Phi = uniqFreqs[ix] * PhiT_invS_Phi
      invTerms[ix] = inv(I + nPhiT_invS_Phi)

    for clz in range(nb_classes):
      nb_samples = cls_counts[clz]
      PhiT_invS_y = np.dot(PhiT_invS, F[clz, :])
      idx = np.flatnonzero(uniqFreqs == nb_samples)[0]
      Cyy = invTerms[idx]
      Ey_clz[clz, :] = np.dot(Cyy, PhiT_invS_y)
      Eyy += nb_samples * Cyy

    Eyy += np.dot((Ey_clz * cls_counts[:, None]).T, Ey_clz)
    return Ey_clz, Eyy

  def compute_llk(self, X):
    """
    Parameters
    ----------
    X : [num_samples, feat_dim]
    """
    nb_samples = X.shape[0]
    S = np.dot(self.Phi, self.Phi.T) + self.Sigma # [feat_dim, feat_dim]
    llk = -0.5 * (self.feat_dim * nb_samples * np.log(2 * np.pi) +
                  nb_samples * logdet(S) +
                  np.sum(X * solve(S, X.T).T))
    return llk

  def maximization_plda(self, X, X_sqr, F, Ey, Eyy):
    """
    ML re-estimation of the Eignevoice subspace and the covariance of the
    residual noise (full).

    Paremters
    ---------
    X : [nb_samples, feat_dim]
    X_cov : [feat_dim, feat_dim]
    F : [nb_classes, feat_dim]
    Ey : [nb_classes, num_phi]
    Eyy : [num_phi, num_phi]
    """
    nb_samples = X.shape[0]
    Ey_FT = np.dot(Ey.T, F) # [num_phi, feat_dim]
    self.Phi = solve(Eyy.T, Ey_FT).T # [feat_dim, num_phi]
    self.Sigma = 1. / nb_samples * (X_sqr - np.dot(self.Phi, Ey_FT))

  def predict_log_proba(self, X, X_model=None):
    """
    Parameters
    ----------
    X : [nb_samples, feat_dim]
    X_model : [nb_classes, feat_dim]
      if None, use class average extracted based on fitted data
    """
    if not self.is_fitted:
      raise RuntimeError("This model hasn't been fitted!")
    # ====== check X_model ====== #
    if X_model is None:
      X_model = self.X_model
    else:
      # [nb_classes, num_phi]
      X_model = np.dot(self.normalizer.transform(X_model), self.Uk)
    if X_model.shape[0] != self.nb_classes:
      raise ValueError("The model matrix contains %d classes, but the "
                       "fitted number of classes is %d" %
                       (X_model.shape[0], self.nb_classes))
    # ====== check X ====== #
    if isinstance(X, (tuple, list)):
      X = np.asarray(X)
    elif "odin.fuel" in str(type(X)):
      X = X[:]
    # ====== transform the input matrices ====== #
    X = np.dot(self.normalizer.transform(X), self.Uk) # [nb_samples, num_phi]
    # [nb_classes, 1]
    score_h1 = np.sum(np.dot(X_model, self.Q_hat) * X_model, axis=1, keepdims=True)
    # [nb_samples, 1]
    score_h2 = np.sum(np.dot(X, self.Q_hat) * X, axis=1, keepdims=True)
    # [nb_samples, nb_classes]
    score_h1h2 = 2 * np.dot(X, np.dot(X_model, self.Lambda).T)
    # [nb_samples, nb_classes]
    scores = score_h1h2 + score_h1.T + score_h2
    return scores
