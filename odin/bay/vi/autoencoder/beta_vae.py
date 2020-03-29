import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class BetaVAE(VariationalAutoencoder):

  def __init__(self, beta=1.0, name='BetaVAE', **kwargs):
    super().__init__(name=name, **kwargs)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    div = self.beta * div
    return llk, div


class AnnealedVAE(VariationalAutoencoder):

  def __init__(self, beta=1.0, name='BetaVAE', **kwargs):
    super().__init__(name=name, **kwargs)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    div = self.beta * div
    return llk, div
