from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TensorTypes
from odin.bay.vi.losses import disentangled_inferred_prior_loss
from odin.utils import as_tuple
from tensorflow import Tensor
from tensorflow_probability.python.distributions import Distribution


class dipVAE(betaVAE):
  r""" Implementation of disentangled infered prior VAE

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
               only_mean: bool = False,
               lambda_diag: float = 1.0,
               lambda_offdiag: float = 2.0,
               beta: float = 1.0,
               **kwargs):
    super().__init__(beta=beta, **kwargs)
    self.only_mean = bool(only_mean)
    self.lambda_diag = tf.convert_to_tensor(lambda_diag,
                                            dtype=self.dtype,
                                            name='lambda_diag')
    self.lambda_offdiag = tf.convert_to_tensor(lambda_offdiag,
                                               dtype=self.dtype,
                                               name='lambda_offdiag')

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    for z, qz in zip(as_tuple(self.latents), as_tuple(qz_x)):
      dip = disentangled_inferred_prior_loss(qz,
                                             only_mean=self.only_mean,
                                             lambda_offdiag=self.lambda_offdiag,
                                             lambda_diag=self.lambda_diag)
      kl[f'dip_{z.name}'] = dip
    return llk, kl
