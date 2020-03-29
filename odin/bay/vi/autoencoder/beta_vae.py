import tensorflow as tf

from odin.backend import interpolation as _interp
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class BetaVAE(VariationalAutoencoder):
  r""" Implementation of beta-VAE
  Arguments:
    beta : a Scalar. A regularizer weight indicate the capacity of the latent.

  Reference:
    Higgins, I., Matthey, L., Pal, A., et al. "beta-VAE: Learning Basic
      Visual Concepts with a Constrained Variational Framework".
      ICLR'17
  """

  def __init__(self, beta=1.0, name='BetaVAE', **kwargs):
    super().__init__(name=name, **kwargs)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    div = self.beta * div
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
               name='AnnealedVAE',
               **kwargs):
    super().__init__(name=name, **kwargs)
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
