import tensorflow as tf

from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.utils import disentangled_prior_loss


class DIPVAE(BetaVAE):
  r""" Implementation of disentangled prior VAE

  Arguments:
    only_mean : A Boolean. If `True`, applying DIP constraint only on the
      mean of latents `Cov[E(z)]` (i.e. type 'i'), otherwise,
      `E[Cov(z)] + Cov[E(z)]` (i.e. type 'ii')
    lambda_offdiag : A Scalar. Weight for penalizing the off-diagonal part of
      covariance matrix.
    lambda_diag : A Scalar. Weight for penalizing the diagonal.

  Reference:
    Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. "Variational Inference
      of Disentangled Latent Concepts from Unlabeled Observations".
      arXiv:1711.00848 [cs, stat].
  """

  def __init__(self,
               only_mean=False,
               lambda_diag=1.0,
               lambda_offdiag=2.0,
               beta=1.0,
               name='DIPVAE',
               **kwargs):
    super().__init__(beta=beta, name=name, **kwargs)
    self.only_mean = bool(only_mean)
    self.lambda_diag = tf.convert_to_tensor(lambda_diag,
                                            dtype=self.dtype,
                                            name='lambda_diag')
    self.lambda_offdiag = tf.convert_to_tensor(lambda_offdiag,
                                               dtype=self.dtype,
                                               name='lambda_offdiag')

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    dip = disentangled_prior_loss(qZ_X,
                                  only_mean=self.only_mean,
                                  lambda_offdiag=self.lambda_offdiag,
                                  lambda_diag=self.lambda_diag)
    div = div + dip
    return llk, div
