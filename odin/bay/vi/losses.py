import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution, Normal

__all__ = [
    'disentangled_inferred_prior_loss',
    'total_correlation',
    'pairwise_distances',
    'gaussian_kernel',
    'maximum_mean_discrepancy',
]


def disentangled_inferred_prior_loss(qZ_X: Distribution,
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
  r"""Estimate of total correlation using Gaussian distribution on a batch.

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


# ===========================================================================
# Maximum-mean discrepancy
# ===========================================================================
def pairwise_distances(x, y, keepdims=True):
  r"""
  Arguments:
    x : a Tensor batch_shape1 + (dim,)
    y : a Tensor batch_shape2 + (dim,)
    keepdims : a Boolean. If True, reshape the output to keep the batch_shape1
      and batch_shape2, otherwise, return flattened output.

  Return:
    distance : a Tensor (batch_shape1, batch_shape2, dim).
      Pairwise distances for each row in x and y
  """
  shape_x = tf.shape(x)
  shape_y = tf.shape(y)
  tf.assert_equal(shape_x[-1], shape_y[-1],
                  "The last dimension of x and y must be equal")
  feat_dim = shape_x[-1]
  # reshape to 2-D
  if tf.rank(x) > 2:
    x = tf.reshape(x, (-1, feat_dim))
  if tf.rank(y) > 2:
    y = tf.reshape(y, (-1, feat_dim))
  # distance
  x = tf.expand_dims(x, axis=1)
  d = x - y
  # reshape to the original
  if keepdims:
    d = tf.reshape(d,
                   tf.concat([shape_x[:-1], shape_y[:-1], (feat_dim,)], axis=0))
  return d


def gaussian_kernel(x, y, sigmas=None):
  r""" Gaussian radial basis function

  Arguments:
    x : a Tensor [num_samples, num_features]
    y : a Tensor [num_samples, num_features]
    sigmas: a Tensor of floats which denote the widths of each of the
      gaussians in the kernel.

  Reference:
    Radial basis function kernel :
      https://en.wikipedia.org/wiki/Radial_basis_function_kernel
  """
  d = pairwise_distances(x, y, keepdims=False)
  if sigmas is None:
    gamma = 1. / tf.cast(tf.shape(x)[-1], dtype=d.dtype)
  else:
    sigmas = tf.convert_to_tensor(sigmas, dtype=d.dtype)
    gamma = 1. / (2. * tf.square(sigmas))
  # L2-norm
  d = tf.reduce_sum(tf.square(d), axis=-1)
  # make sure gamma is broadcastable
  if tf.rank(gamma) == 0:
    gamma = tf.expand_dims(gamma, axis=0)
  for _ in tf.range(tf.rank(d)):
    gamma = tf.expand_dims(gamma, axis=0)
  return tf.reduce_sum(tf.math.exp(-tf.expand_dims(d, axis=-1) * gamma),
                       axis=-1)


def linear_kernel(x, y):
  d = pairwise_distances(x, y, keepdims=False)
  return tf.math.abs(tf.reduce_sum(d, axis=-1))


def polynomial_kernel(x, y, d=2):
  d = pairwise_distances(x, y, keepdims=False)
  return tf.math.abs(tf.reduce_sum(d, axis=-1))


def maximum_mean_discrepancy(qZ, pZ, nq=10, np=10, seed=1, kernel='gaussian'):
  r""" is a distance-measure between distributions p(X) and q(Y) which is
  defined as the squared distance between their embeddings in the a
  "reproducing kernel Hilbert space".

  Given n examples from p(X) and m samples from q(Y), one can formulate a
  test statistic based on the empirical estimate of the MMD:

  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }

  Arguments:
    nq : a Scalar. Number of posterior samples
    np : a Scalar. Number of prior samples

  Reference:
    Gretton, A., Borgwardt, K., Rasch, M.J., Scholkopf, B., Smola, A.J., 2008.
      "A Kernel Method for the Two-Sample Problem". arXiv:0805.2368 [cs].

  """
  assert isinstance(
      qZ, Distribution
  ), 'qZ must be instance of tensorflow_probability.Distribution'
  assert isinstance(
      pZ, Distribution
  ), 'pZ must be instance of tensorflow_probability.Distribution'
  x = qZ.sample(int(nq), seed=seed)
  y = pZ.sample(int(np), seed=seed)
  if kernel == 'gaussian':
    kernel = gaussian_kernel
  elif kernel == 'linear':
    kernel = linear_kernel
  elif kernel == 'polynomial':
    kernel = polynomial_kernel
  else:
    raise NotImplementedError("No support for kernel: '%s'" % kernel)
  k_xx = kernel(x, x)
  k_yy = kernel(y, y)
  k_xy = kernel(x, y)
  return tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)
