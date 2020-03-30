import tensorflow as tf

from odin.backend import interpolation as _interp
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.losses import total_correlation


class BetaVAE(VariationalAutoencoder):
  r""" Implementation of beta-VAE
  Arguments:
    beta : a Scalar. A regularizer weight indicate the capacity of the latent.

  Reference:
    Higgins, I., Matthey, L., Pal, A., et al. "beta-VAE: Learning Basic
      Visual Concepts with a Constrained Variational Framework".
      ICLR'17
  """

  def __init__(self, beta=1.0, **kwargs):
    super().__init__(**kwargs)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    div = self.beta * div
    return llk, div


class BetaTCVAE(BetaVAE):
  r""" Extend the beta-VAE with total correlation loss added.

  Based on Equation (4) with alpha = gamma = 1
  If alpha = gamma = 1, Eq. 4 can be written as `ELBO + (1 - beta) * TC`.

  Reference:
    Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. "Isolating Sources
      of Disentanglement in Variational Autoencoders".
      arXiv:1802.04942 [cs, stat].
  """

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    tc = tf.constant(0., dtype=div.dtype)
    for q in tf.nest.flatten(qZ_X):
      tc += total_correlation(tf.convert_to_tensor(q), q)
    div += (self.beta - 1.) * tc
    return llk, div


class AnnealedVAE(VariationalAutoencoder):
  r"""Creates an AnnealedVAE model.

  Implementing Eq. 8 of (Burgess et al. 2018)

  Arguments:
    gamma: Hyperparameter for the regularizer.
    c_max: a Scalar. Maximum capacity of the bottleneck.
      is gradually increased from zero to a value large enough to produce
      good quality reconstructions
    iter_max: an Integer. Number of iteration until reach the maximum
      capacity (start from 0).
    interpolation : a String. Type of interpolation for increasing capacity.

  Example:
    vae = AnnealedVAE()
    elbo = vae.elbo(x, px, qz, n_iter=1)

  Reference:
    Burgess, C.P., Higgins, I., et al. 2018. "Understanding disentangling in
      beta-VAE". arXiv:1804.03599 [cs, stat].
  """

  def __init__(self,
               gamma=1.0,
               c_max=100,
               iter_max=100,
               interpolation='linear',
               **kwargs):
    super().__init__(**kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.c_max = tf.convert_to_tensor(c_max, dtype=self.dtype, name='c_max')
    self.interpolation = _interp.get(str(interpolation))(
        vmin=tf.convert_to_tensor(0, self.dtype),
        vmax=self.c_max,
        norm=int(iter_max))

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc, n_iter=0):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    c = self.interpolation(n_iter)
    div = self.gamma * tf.math.abs(div - c)
    return llk, div
