# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
from scipy import linalg

from multiprocessing import Value, Array

from sklearn.decomposition import IncrementalPCA
from sklearn.utils import check_array, gen_batches
from sklearn.utils.extmath import svd_flip, _incremental_mean_and_var, fast_dot

from odin.utils.mpi import MPI
from odin.utils import batching
from odin.fuel import Data

__all__ = [
    "MiniBatchPCA"
]


class MiniBatchPCA(IncrementalPCA):
  """ A modified version of IncrementalPCA to effectively
  support multi-processing (but not work)
  Original Author: Kyle Kastner <kastnerkyle@gmail.com>
                   Giorgio Patrini
  License: BSD 3 clause

  Incremental principal components analysis (IPCA).

  Linear dimensionality reduction using Singular Value Decomposition of
  centered data, keeping only the most significant singular vectors to
  project the data to a lower dimensional space.

  Depending on the size of the input data, this algorithm can be much more
  memory efficient than a PCA.

  This algorithm has constant memory complexity, on the order
  of ``batch_size``, enabling use of np.memmap files without loading the
  entire file into memory.

  The computational overhead of each SVD is
  ``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
  remain in memory at a time. There will be ``n_samples / batch_size`` SVD
  computations to get the principal components, versus 1 large SVD of
  complexity ``O(n_samples * n_features ** 2)`` for PCA.

  Read more in the :ref:`User Guide <IncrementalPCA>`.

  Parameters
  ----------
  n_components : int or None, (default=None)
      Number of components to keep. If ``n_components `` is ``None``,
      then ``n_components`` is set to ``min(n_samples, n_features)``.

  batch_size : int or None, (default=None)
      The number of samples to use for each batch. Only used when calling
      ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
      is inferred from the data and set to ``5 * n_features``, to provide a
      balance between approximation accuracy and memory consumption.

  copy : bool, (default=True)
      If False, X will be overwritten. ``copy=False`` can be used to
      save memory but is unsafe for general use.

  whiten : bool, optional
      When True (False by default) the ``components_`` vectors are divided
      by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
      with unit component-wise variances.

      Whitening will remove some information from the transformed signal
      (the relative variance scales of the components) but can sometimes
      improve the predictive accuracy of the downstream estimators by
      making data respect some hard-wired assumptions.

  Attributes
  ----------
  components_ : array, shape (n_components, n_features)
      Components with maximum variance.

  explained_variance_ : array, shape (n_components,)
      Variance explained by each of the selected components.

  explained_variance_ratio_ : array, shape (n_components,)
      Percentage of variance explained by each of the selected components.
      If all components are stored, the sum of explained variances is equal
      to 1.0

  mean_ : array, shape (n_features,)
      Per-feature empirical mean, aggregate over calls to ``partial_fit``.

  var_ : array, shape (n_features,)
      Per-feature empirical variance, aggregate over calls to
      ``partial_fit``.

  noise_variance_ : float
      The estimated noise covariance following the Probabilistic PCA model
      from Tipping and Bishop 1999. See "Pattern Recognition and
      Machine Learning" by C. Bishop, 12.2.1 p. 574 or
      http://www.miketipping.com/papers/met-mppca.pdf.

  n_components_ : int
      The estimated number of components. Relevant when
      ``n_components=None``.

  n_samples_seen_ : int
      The number of samples processed by the estimator. Will be reset on
      new calls to fit, but increments across ``partial_fit`` calls.

  Notes
  -----
  Implements the incremental PCA model from:
  `D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
  Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
  pp. 125-141, May 2008.`
  See http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

  This model is an extension of the Sequential Karhunen-Loeve Transform from:
  `A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis Extraction and
  its Application to Images, IEEE Transactions on Image Processing, Volume 9,
  Number 8, pp. 1371-1374, August 2000.`
  See http://www.cs.technion.ac.il/~mic/doc/skl-ip.pdf

  We have specifically abstained from an optimization used by authors of both
  papers, a QR decomposition used in specific situations to reduce the
  algorithmic complexity of the SVD. The source for this technique is
  `Matrix Computations, Third Edition, G. Holub and C. Van Loan, Chapter 5,
  section 5.4.4, pp 252-253.`. This technique has been omitted because it is
  advantageous only when decomposing a matrix with ``n_samples`` (rows)
  >= 5/3 * ``n_features`` (columns), and hurts the readability of the
  implemented algorithm. This would be a good opportunity for future
  optimization, if it is deemed necessary.

  For `multiprocessing`, you can do parallelized `partial_fit` or `transform`
  but you cannot do `partial_fit` in one process and `transform` in the others.

  Application
  -----------
  In detail, in order for PCA to work well, informally we require that
  (i) The features have approximately zero mean, and
  (ii) The different features have similar variances to each other.
  With natural images, (ii) is already satisfied even without variance
  normalization, and so we won’t perform any variance normalization.
  (If you are training on audio data—say, on spectrograms—or on text data—say,
  bag-of-word vectors—we will usually not perform variance normalization
  either.)

  By using PCA, we aim for:
  (i) the features are less correlated with each other, and
  (ii) the features all have the same variance.


  Original link: http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/

  References
  ----------
  D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
      Tracking, International Journal of Computer Vision, Volume 77,
      Issue 1-3, pp. 125-141, May 2008.

  G. Golub and C. Van Loan. Matrix Computations, Third Edition, Chapter 5,
      Section 5.4.4, pp. 252-253.

  See also
  --------
  PCA
  RandomizedPCA
  KernelPCA
  SparsePCA
  TruncatedSVD
  """

  def __init__(self, n_components=None, whiten=False, copy=True,
               batch_size=None):
    super(MiniBatchPCA, self).__init__(n_components=n_components,
        whiten=whiten, copy=copy, batch_size=batch_size)
    # some statistics
    self.n_samples_seen_ = 0
    self.mean_ = .0
    self.var_ = .0
    self.components_ = None
    # if nb_samples < nb_components, then the mini batch is cached until
    # we have enough samples
    self._cache_batches = []
    self._nb_cached_samples = 0

  @property
  def is_fitted(self):
    return self.components_ is not None

  # ==================== Training ==================== #
  def fit(self, X, y=None):
    """Fit the model with X, using minibatches of size batch_size.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
        n_features is the number of features.

    y: Passthrough for ``Pipeline`` compatibility.

    Returns
    -------
    self: object
        Returns the instance itself.
    """
    if isinstance(X, Data):
      X = X[:]
    X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])
    n_samples, n_features = X.shape

    if self.batch_size is None:
      batch_size = 12 * n_features
    else:
      batch_size = self.batch_size

    for batch in gen_batches(n_samples, batch_size):
      x = X[batch]
      self.partial_fit(x, check_input=False)
    return self

  def partial_fit(self, X, y=None, check_input=True):
    """Incremental fit with X. All of X is processed as a single batch.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
        n_features is the number of features.

    Returns
    -------
    self: object
        Returns the instance itself.
    """
    # ====== check the samples and cahces ====== #
    if isinstance(X, Data):
      X = X[:]
    if check_input:
      X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])
    n_samples, n_features = X.shape
    # check number of components
    if self.n_components is None:
      self.n_components_ = n_features
    elif not 1 <= self.n_components <= n_features:
      raise ValueError("n_components=%r invalid for n_features=%d, need "
                       "more rows than columns for IncrementalPCA "
                       "processing" % (self.n_components, n_features))
    else:
      self.n_components_ = self.n_components
    # check the cache
    if n_samples < n_features or self._nb_cached_samples > 0:
      self._cache_batches.append(X)
      self._nb_cached_samples += n_samples
      # not enough samples yet
      if self._nb_cached_samples < n_features:
        return
      else: # group mini batch into big batch
        X = np.concatenate(self._cache_batches, axis=0)
        self._cache_batches = []
        self._nb_cached_samples = 0
    n_samples = X.shape[0]
    # ====== fit the model ====== #
    if (self.components_ is not None) and (self.components_.shape[0] !=
                                           self.n_components_):
      raise ValueError("Number of input features has changed from %i "
                       "to %i between calls to partial_fit! Try "
                       "setting n_components to a fixed value." %
                       (self.components_.shape[0], self.n_components_))
    # Update stats - they are 0 if this is the fisrt step
    col_mean, col_var, n_total_samples = \
        _incremental_mean_and_var(X, last_mean=self.mean_,
                                  last_variance=self.var_,
                                  last_sample_count=self.n_samples_seen_)
    total_var = np.sum(col_var * n_total_samples)
    if total_var == 0: # if variance == 0, make no sense to continue
      return self
    # Whitening
    if self.n_samples_seen_ == 0:
      # If it is the first step, simply whiten X
      X -= col_mean
    else:
      col_batch_mean = np.mean(X, axis=0)
      X -= col_batch_mean
      # Build matrix of combined previous basis and new data
      mean_correction = \
          np.sqrt((self.n_samples_seen_ * n_samples) /
                  n_total_samples) * (self.mean_ - col_batch_mean)
      X = np.vstack((self.singular_values_.reshape((-1, 1)) *
                    self.components_, X, mean_correction))

    U, S, V = linalg.svd(X, full_matrices=False)
    U, V = svd_flip(U, V, u_based_decision=False)
    explained_variance = S ** 2 / n_total_samples
    explained_variance_ratio = S ** 2 / total_var

    self.n_samples_seen_ = n_total_samples
    self.components_ = V[:self.n_components_]
    self.singular_values_ = S[:self.n_components_]
    self.mean_ = col_mean
    self.var_ = col_var
    self.explained_variance_ = explained_variance[:self.n_components_]
    self.explained_variance_ratio_ = \
        explained_variance_ratio[:self.n_components_]
    if self.n_components_ < n_features:
      self.noise_variance_ = \
          explained_variance[self.n_components_:].mean()
    else:
      self.noise_variance_ = 0.
    return self

  def transform(self, X, n_components=None):
    # ====== check number of components ====== #
    # specified percentage of explained variance
    if n_components is not None:
      # percentage of variances
      if n_components < 1.:
        _ = np.cumsum(self.explained_variance_ratio_)
        n_components = (_ > n_components).nonzero()[0][0] + 1
      # specific number of components
      else:
        n_components = int(n_components)
    # ====== other info ====== #
    n = X.shape[0]
    if self.batch_size is None:
      batch_size = 12 * len(self.mean_)
    else:
      batch_size = self.batch_size
    # ====== start transforming ====== #
    X_transformed = []
    for start, end in batching(n=n, batch_size=batch_size):
      x = super(MiniBatchPCA, self).transform(X=X[start:end])
      if n_components is not None:
        x = x[:, :n_components]
      X_transformed.append(x)
    return np.concatenate(X_transformed, axis=0)

  def invert_transform(self, X):
    if isinstance(X, Data):
      X = X[:]
    return super(MiniBatchPCA, self).inverse_transform(X=X)

  def transform_mpi(self, X, keep_order=True, ncpu=4,
                    n_components=None):
    """ Sample as transform but using multiprocessing """
    n = X.shape[0]
    if self.batch_size is None:
      batch_size = 12 * len(self.mean_)
    else:
      batch_size = self.batch_size
    batch_list = [(i, min(i + batch_size, n))
        for i in range(0, n + batch_size, batch_size) if i < n]

    # ====== run MPI jobs ====== #
    def map_func(batch):
      start, end = batch
      x = super(MiniBatchPCA, self).transform(X=X[start:end])
      # doing dim reduction here save a lot of memory for
      # inter-processors transfer
      if n_components is not None:
        x = x[:, :n_components]
      # just need to return the start for ordering
      yield start, x
    mpi = MPI(batch_list, func=map_func,
              ncpu=ncpu, batch=1, hwm=ncpu * 12,
              backend='python')
    # ====== process the return ====== #
    X_transformed = []
    for start, x in mpi:
      X_transformed.append((start, x))
    if keep_order:
      X_transformed = sorted(X_transformed, key=lambda x: x[0])
    X_transformed = np.concatenate([x[-1] for x in X_transformed], axis=0)
    return X_transformed
