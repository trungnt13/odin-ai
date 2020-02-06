from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution, Normal


# ===========================================================================
# Metrics
# ===========================================================================
def discretizing(*factors,
                 independent=True,
                 n_bins=5,
                 strategy='quantile',
                 encode='ordinal',
                 return_model=False):
  r""" Transform continuous value into discrete

  Note: the histogram discretizer is equal to
    `KBinsDiscretizer(n_bins=., encode='ordinal', strategy='uniform')`

  Arguments:
    factors : array-like or list of array-like
    independent : a Boolean, if `True` (by default), each factor is
      discretize independently.
    n_bins : int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.
    encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='ordinal')
        Method used to encode the transformed result.
        onehot - Encode the transformed result with one-hot encoding and
          return a sparse matrix. Ignored features are always stacked to
          the right.
        onehot-dense - Encode the transformed result with one-hot encoding
          and return a dense array. Ignored features are always stacked to
          the right.
        ordinal - Return the bin identifier encoded as an integer value.
    strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
        Strategy used to define the widths of the bins.
        uniform - All bins in each feature have identical widths.
        quantile - All bins in each feature have the same number of points.
        kmeans - Values in each bin have the same nearest center of a 1D
          k-means cluster.
  """
  from sklearn.preprocessing import KBinsDiscretizer
  strategy = str(strategy).strip().lower()
  encode = str(encode).strip().lower()
  disc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
  if independent:
    disc.fit(factors[0])
    transform = lambda x: disc.transform(x).astype(np.int64)
  else:
    disc.fit(np.expand_dims(factors[0].ravel(), axis=-1))
    transform = lambda x: np.hstack([
        disc.transform(np.expand_dims(i, axis=-1)).astype(np.int64) for i in x.T
    ])
  factors = tuple([transform(i) for i in factors])
  factors = factors[0] if len(factors) == 1 else factors
  if return_model:
    return factors, disc
  return factors


def discrete_mutual_info(codes, factors):
  r"""Compute discrete mutual information.

  Arguments:
    codes : `[n_samples, n_codes]`, the latent codes or predictive codes
    factors : `[n_samples, n_factors]`, the groundtruth factors

  Return:
    matrix `[n_codes, n_factors]` : mutual information score between factor
      and code
  """
  from sklearn.metrics import mutual_info_score
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
  r"""Compute discrete entropy.

  Arguments:
    labels : 1-D or 2-D array

  Returns:
    entropy : A Scalar or array `[n_factors]`
  """
  from sklearn.metrics.cluster import entropy
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
  from sklearn.ensemble import GradientBoostingClassifier
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
  return d, c, test_acc


# ===========================================================================
# Others
# ===========================================================================
def permute_dims(z):
  r""" Permutation of latent dimensions Algorithm(1)

  Arguments:
    z : A Tensor `[batch_size, latent_dim]`

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """
  batch_dim, latent_dim = z.shape
  perm = tf.TensorArray(dtype=z.dtype,
                        size=latent_dim,
                        dynamic_size=False,
                        clear_after_read=False,
                        element_shape=(batch_dim,))
  ids = tf.range(z.shape[0], dtype=tf.int32)
  # iterate over latent dimension
  for i in tf.range(latent_dim):
    # shuffle among minibatch
    z_i = tf.gather(z[:, i], tf.random.shuffle(ids))
    perm = perm.write(i, z_i)
  return tf.transpose(perm.stack())


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
