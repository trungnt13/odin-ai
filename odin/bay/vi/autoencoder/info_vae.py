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
    Shengjia Zhao. "A Tutorial on Information Maximizing Variational
      Autoencoders (InfoVAE)".
      https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders
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
    # select right divergence
    if self.divergence == 'mmd':
      fn = maximum_mean_discrepancy
    else:
      raise NotImplementedError("No support for divergence family: '%s'" %
                                self.divergence)
    # repeat for each latent
    d = tf.constant(0., dtype=div.dtype)
    for q in tf.nest.flatten(qZ_X):
      d += fn(qZ=q, pZ=q.KL_divergence.prior, **self.divergence_kw)
    return llk, div + (self.gamma - self.beta) * d


class IFVAE(BetaVAE):
  r""" Adversarial information factorized VAE

  Reference:
    Creswell, A., Mohamied, Y., Sengupta, B., Bharath, A.A., 2018.
      "Adversarial Information Factorization". arXiv:1711.05175 [cs].
  """
  pass
