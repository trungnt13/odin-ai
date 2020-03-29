from __future__ import absolute_import, division, print_function

import types

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
from tensorflow_probability.python.distributions import Distribution, Normal

__all__ = [
    'discretizing',
    'permute_dims',
    'dip_loss',
    'total_correlation',
]


def _gmm_discretizing_predict(self, X):
  self._check_is_fitted()
  means = self.means_.ravel()
  ids = self._estimate_weighted_log_prob(X).argmax(axis=1)
  # sort by increasing order of means_
  return np.expand_dims(np.argsort(means)[ids], axis=1)


def discretizing(*factors,
                 independent=True,
                 n_bins=5,
                 strategy='quantile',
                 return_model=False):
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


def dip_loss(qZ_X: Distribution,
             only_mean=False,
             lambda_offdiag=2.,
             lambda_diag=1.):
  r""" Disentangled inferred prior (DIP) matches the covariance of the prior
  distributions with the inferred prior

  Uses `cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T`.

  Arguments:
    qZ_X : `tensorflow_probability.Distribution`
    only_mean : A Boolean. If `True`, applying DIP constraint only on the
      mean of latents `Cov[E(z)]` (i.e. type 'i'),
      otherwise, `E[Cov(z)] + Cov[E(z)]` (i.e. type 'ii')
    lambda_offdiag : A Scalar. Weight for penalizing the off-diagonal part of
      covariance matrix.
    lambda_diag : A Scalar. Weight for penalizing the diagonal.

  Reference:
    Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. Variational Inference of
      Disentangled Latent Concepts from Unlabeled Observations.
      arXiv:1711.00848 [cs, stat].
    Github code https://github.com/IBM/AIX360
    Github code https://github.com/google-research/disentanglement_lib

  """
  z_mean = qZ_X.mean()
  shape = z_mean.shape
  if len(shape) > 2:
    # [n_mcmc * batch_size, zdim]
    z_mean = tf.reshape(
        z_mean, (tf.cast(tf.reduce_prod(shape[:-1]), tf.int32),) + shape[-1:])
  expectation_z_mean_z_mean_t = tf.reduce_mean(tf.expand_dims(z_mean, 2) *
                                               tf.expand_dims(z_mean, 1),
                                               axis=0)
  expectation_z_mean = tf.reduce_mean(z_mean, axis=0)
  # cov_zmean [zdim, zdim]
  cov_zmean = tf.subtract(
      expectation_z_mean_z_mean_t,
      tf.expand_dims(expectation_z_mean, 1) *
      tf.expand_dims(expectation_z_mean, 0))
  # Eq(5)
  if only_mean:
    z_cov = cov_zmean
  else:
    z_var = qZ_X.variance()
    if len(shape) > 2:
      z_var = tf.reshape(
          z_var, (tf.cast(tf.reduce_prod(shape[:-1]), tf.int32),) + shape[-1:])
    # mean_zcov [zdim, zdim]
    mean_zcov = tf.reduce_mean(tf.linalg.diag(z_var), axis=0)
    z_cov = cov_zmean + mean_zcov
  # Eq(6) and Eq(7)
  # z_cov [n_mcmc, zdim, zdim]
  # z_cov_diag [n_mcmc, zdim]
  # z_cov_offdiag [n_mcmc, zdim, zdim]
  z_cov_diag = tf.linalg.diag_part(z_cov)
  z_cov_offdiag = z_cov - tf.linalg.diag(z_cov_diag)
  return lambda_offdiag * tf.reduce_sum(z_cov_offdiag ** 2) + \
    lambda_diag * tf.reduce_sum((z_cov_diag - 1.) ** 2)


def total_correlation(z_samples, qZ_X: Distribution):
  r"""Estimate of total correlation on a batch.

  We need to compute the expectation over a batch of:
  `E_j [log(q(z(x_j))) - log(prod_l q(z(x_j)_l))]`

  We ignore the constants as they do not matter for the minimization.
  The constant should be equal to
  `(num_latents - 1) * log(batch_size * dataset_size)`

  If `alpha = gamma = 1`, Eq(4) can be written as `ELBO + (1 - beta) * TC`.
  (i.e. `(1. - beta) * total_correlation(z_sampled, qZ_X)`)

  Arguments:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.

  Returns:
    Total correlation estimated on a batch.

  Reference:
    Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. Isolating Sources of
      Disentanglement in Variational Autoencoders. arXiv:1802.04942 [cs, stat].
    Github code https://github.com/google-research/disentanglement_lib
  """
  gaus = Normal(loc=tf.expand_dims(qZ_X.mean(), 0),
                scale=tf.expand_dims(qZ_X.stddev(), 0))
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaus.log_prob(tf.expand_dims(z_samples, 1))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob,
                                                     axis=1,
                                                     keepdims=False),
                                 axis=1,
                                 keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob,
                                             axis=2,
                                             keepdims=False),
                               axis=1,
                               keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)
