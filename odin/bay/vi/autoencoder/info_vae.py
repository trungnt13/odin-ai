from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import (Distribution,
                                                         OneHotCategorical)

from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.discriminator import FactorDiscriminator
from odin.bay.vi.losses import get_divergence


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
    fn = get_divergence(self.divergence)
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


class InfoMaxVAE(BetaVAE):
  r"""
  Reference:
    Rezaabad, A.L., Vishwanath, S., 2020. "Learning Representations by
      Maximizing Mutual Information in Variational Autoencoders".
      arXiv:1912.13361 [cs, stat].
    Hjelm, R.D., Fedorov, A., et al. 2019. "Learning Deep Representations by
      Mutual Information Estimation and Maximization". ICLR'19.
  """

  def __init__(self, beta=1.0, alpha=1.0, **kwargs):
    super().__init__(beta=beta, **kwargs)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    return llk, div


class MutualInfoVAE(BetaVAE):
  r"""
  Reference:
    Ducau, F.N., Trénous, S. "Mutual Information in Variational Autoencoders".
      https://github.com/fducau/infoVAE.
    Chen, X., Chen, X., Duan, Y., et al. (2016) "InfoGAN: Interpretable
      Representation Learning by Information Maximizing Generative
      Adversarial Nets". URL : http://arxiv.org/ abs/1606.03657.
    Ducau, F.N. Code:  https://github.com/fducau/infoVAE
  """

  def __init__(self,
               beta=1.0,
               gamma=1.0,
               latents=RandomVariable(event_shape=10,
                                      posterior='diag',
                                      name="LatentVariable"),
               code=RandomVariable(event_shape=10,
                                   posterior='diag',
                                   name='Code'),
               **kwargs):
    latents = tf.nest.flatten(latents)
    latents.append(code)
    self.is_binary_code = code.is_binary
    super().__init__(beta=beta,
                     latents=latents,
                     reduce_latent='concat',
                     **kwargs)
    self.code = self.latent_layers[-1]
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc, training=None):
    # don't take KL of qC_X
    llk, div = super()._elbo(X, pX_Z, qZ_X[:-1], analytic, reverse, n_mcmc)
    # the latents
    z = tf.concat([q.sample() for q in qZ_X[:-1]], axis=-1)
    # mutual information code
    qC_X = qZ_X[-1]
    c_prime = qC_X.KL_divergence.prior.sample(z.shape[:-1])
    if self.is_binary_code:
      c_prime = tf.clip_by_value(c_prime, 1e-8, 1. - 1e-8)
    # decoding
    z_prime = tf.concat([z, c_prime], axis=-1)
    pX_Zprime = self.decode(z_prime, training=training)
    qC_Xprime = self.encode(pX_Zprime, training=training)[-1]
    # mutual information
    mi = qC_Xprime.log_prob(c_prime)
    return llk + self.gamma * mi, div


# class AuxInfoVAE(BetaVAE):
#   r""" Maximizing mutual information VAE via auxliary code

#   Reference:
#     Chen, X., Chen, X., Duan, Y., et al. (2016) "InfoGAN: Interpretable
#       Representation Learning by Information Maximizing Generative
#       Adversarial Nets". URL : http://arxiv.org/ abs/1606.03657.
#     Creswell, A., Mohamied, Y., Sengupta, B., Bharath, A.A., (2018).
#       "Adversarial Information Factorization". arXiv:1711.05175 [cs].
#   """

#   def __init__(self,
#                beta=1.0,
#                gamma=1.0,
#                latents=RandomVariable(event_shape=10,
#                                       posterior='diag',
#                                       name="Latent"),
#                code=RandomVariable(event_shape=10,
#                                    posterior='diag',
#                                    name='Code'),
#                learnable_code=False,
#                **kwargs):
#     self.learnable_code = bool(learnable_code)
#     latents = tf.nest.flatten(latents)
#     if self.learnable_code:
#       latents.append(code)
#     super().__init__(beta=beta,
#                      latents=latents,
#                      reduce_latent='concat',
#                      **kwargs)
#     self.code = self.latent_layers[-1]
#     self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype)

#   def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc, training=None):
#     # don't take KL of qC_X
#     llk, div = super()._elbo(X, pX_Z, qZ_X[:-1], analytic, reverse, n_mcmc)
#     return llk, div
