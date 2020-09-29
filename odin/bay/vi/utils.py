from __future__ import absolute_import, division, print_function

import types
import warnings
from numbers import Number
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
from typing_extensions import Literal

__all__ = [
    'discretizing',
    'permute_dims',
    'marginalize_categorical_labels',
]


def _gmm_discretizing_predict(self, X):
  self._check_is_fitted()
  means = self.means_.ravel()
  ids = self._estimate_weighted_log_prob(X).argmax(axis=1)
  # sort by increasing order of means_
  return np.expand_dims(np.argsort(means)[ids], axis=1)


def discretizing(
    *factors: List[np.ndarray],
    independent: bool = True,
    n_bins: int = 5,
    strategy: Literal['uniform', 'quantile', 'kmeans', 'gmm'] = 'quantile',
    return_model: bool = False,
):
  r""" Transform continuous value into discrete

  Note: the histogram discretizer is equal to
    `KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='uniform')`

  Arguments:
    factors : array-like or list of array-like
    independent : a Boolean, if `True` (by default), each factor (i.e. column)
      is discretize independently.
    n_bins : int or array-like, shape (n_features,) (default=5)
      The number of bins to produce. Raises ValueError if ``n_bins < 2``.
    strategy : {'uniform', 'quantile', 'kmeans', 'gmm'}, (default='quantile')
      Strategy used to define the widths of the bins.
      uniform - All bins in each feature have identical widths.
      quantile - All bins in each feature have the same number of points.
      kmeans - Values in each bin have the same nearest center of a 1D
        k-means cluster.
      gmm - using the components (in sorted order of mean) of Gaussian
        mixture to label.
  """
  encode = 'ordinal'
  # onehot - sparse matrix of one-hot encoding and
  # onehot-dense - dense one-hot encoding. Ignored features are always stacked to
  #   the right.
  # ordinal - Return the bin identifier encoded as an integer value.
  strategy = str(strategy).strip().lower()
  if 'histogram' in strategy:
    strategy = 'uniform'
  # ====== GMM base discretizer ====== #
  if 'gmm' in strategy:
    create_gmm = lambda: GaussianMixture(n_components=n_bins,
                                         max_iter=800,
                                         covariance_type='diag',
                                         random_state=1)  # fix random state

    if independent:
      gmm = []
      for f in factors[0].T:
        gm = create_gmm()
        gm.fit(np.expand_dims(f, axis=1))
        gm.predict = types.MethodType(_gmm_discretizing_predict, gm)
        gmm.append(gm)
      transform = lambda x: np.concatenate([
          gm.predict(np.expand_dims(col, axis=1)) for gm, col in zip(gmm, x.T)
      ],
                                           axis=1)
    else:
      gmm = create_gmm()
      gmm.fit(np.expand_dims(factors[0].ravel(), axis=1))
      gmm.predict = types.MethodType(_gmm_discretizing_predict, gmm)
      transform = lambda x: np.concatenate(
          [gmm.predict(np.expand_dims(col, axis=1)) for col in x.T], axis=1)
    disc = gmm
  # ====== start with bins discretizer ====== #
  else:
    disc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    if independent:
      disc.fit(factors[0])
      transform = lambda x: disc.transform(x).astype(np.int64)
    else:
      disc.fit(np.expand_dims(factors[0].ravel(), axis=-1))
      transform = lambda x: np.hstack([
          disc.transform(np.expand_dims(i, axis=-1)).astype(np.int64)
          for i in x.T
      ])
  # ====== returns ====== #
  factors = tuple([transform(i) for i in factors])
  factors = factors[0] if len(factors) == 1 else factors
  if return_model:
    return factors, disc
  return factors


def marginalize_categorical_labels(batch_size, num_classes, dtype=tf.float32):
  r"""
  Example:
  ```
  # shape: [batch_size * n_labels, n_labels]
  y = marginalize_categorical_labels(batch_size=inputs[0].shape[0],
                                     num_classes=n_labels,
                                     dtype=self.dtype)
  # shape: [batch_size * n_labels, n_dims]
  X = [tf.repeat(i, n_labels, axis=0) for i in inputs]
  ```
  """
  y = tf.expand_dims(tf.eye(num_classes, dtype=dtype), axis=0)
  y = tf.repeat(y, batch_size, axis=0)
  y = tf.reshape(y, (-1, num_classes))
  return y


@tf.function(autograph=True)
def permute_dims(z):
  r""" Permutation of latent dimensions Algorithm(1):

  ```
    input: matrix-(batch_dim, latent_dim)
    output: matrix-(batch_dim, latent_dim)

    foreach latent_dim:
      shuffle points along batch_dim
  ```


  Arguments:
    z : A Tensor `[batch_size, latent_dim]`

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """
  shape = z.shape
  batch_dim, latent_dim = shape[-2:]
  perm = tf.TensorArray(dtype=z.dtype,
                        size=latent_dim,
                        dynamic_size=False,
                        clear_after_read=False,
                        element_shape=shape[:-1])
  ids = tf.range(batch_dim, dtype=tf.int32)
  # iterate over latent dimension
  for i in tf.range(latent_dim):
    # shuffle among minibatch
    z_i = tf.gather(z[..., i], tf.random.shuffle(ids), axis=-1)
    perm = perm.write(i, z_i)
  return tf.transpose(perm.stack(),
                      perm=tf.concat([tf.range(1, tf.rank(z)), (0,)], axis=0))
