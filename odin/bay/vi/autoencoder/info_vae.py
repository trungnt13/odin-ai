import tensorflow as tf

from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.losses import maximum_mean_discrepancy


class InfoVAE(BetaVAE):
  r"""
  Arguments:
    alpha : a Scalar. Equal to `1 - beta`
    gamma : a Scalar. This is the value of lambda in the paper
    divergence : a String. Divergences families, for now only support 'mmd'
      i.e. maximum-mean discrepancy.

  Reference:
    Zhao, S., Song, J., Ermon, S., et al. "InfoVAE: Balancing Learning and
      Inference in Variational Autoencoders".
  """

  def __init__(self,
               alpha=0.0,
               gamma=1.0,
               divergence='mmd',
               divergence_kw=dict(kernel='gaussian', nq=10, np=10),
               **kwargs):
    super().__init__(beta=1 - alpha, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.divergence = str(divergence)
    self.divergence_kw = dict(divergence_kw)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    if self.divergence == 'mmd':
      d = maximum_mean_discrepancy(qZ=qZ_X,
                                   pZ=qZ_X.KL_divergence.prior,
                                   **self.divergence_kw)
    else:
      raise NotImplementedError("No support for divergence family: '%s'" %
                                self.divergence)
    return llk, div + (self.gamma - self.beta) * d


class IFVAE(BetaVAE):
  r"""

  Reference:
    Creswell, A., Mohamied, Y., Sengupta, B., Bharath, A.A., 2018.
      "Adversarial Information Factorization". arXiv:1711.05175 [cs].
  """
  pass
