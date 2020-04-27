from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import (Distribution,
                                                         OneHotCategorical)

from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.losses import get_divergence, maximum_mean_discrepancy
from odin.bay.vi.utils import permute_dims


class InfoVAE(BetaVAE):
  r"""
  For MNIST, the authors used scaling coefficient lambda(gamma)=1000, and
  information preference alpha=0.

  Increase `np` (number of prior samples) in `divergence_kw` to reduce the
  variance of MMD estimation.

  Arguments:
    alpha : a Scalar. Equal to `1 - beta`
      Higher value of alpha places lower weight on the KL-divergence
    gamma : a Scalar. This is the value of lambda in the paper
      Higher value of gamma place more weight on the Info-divergence (i.e. MMD)
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
               gamma=100.0,
               divergence='mmd',
               divergence_kw=dict(kernel='gaussian',
                                  q_sample_shape=None,
                                  p_sample_shape=100),
               **kwargs):
    super().__init__(beta=1 - alpha, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    # select right divergence
    self.divergence_name = str(divergence)
    self.divergence = get_divergence(self.divergence_name)
    self.divergence_kw = dict(divergence_kw)

  @property
  def alpha(self):
    return 1 - self.beta

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    llk, div = super()._elbo(X,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training)
    # repeat for each latent
    for name, q in zip(self.latent_names, qZ_X):
      info_div = (self.gamma - self.beta) * self.divergence(
          qZ=q,
          pZ=q.KL_divergence.prior,
          **self.divergence_kw,
      )
      div['%s_%s' % (self.divergence_name, name)] = info_div
    return llk, div


class InfoNCEVAE(BetaVAE):
  r""" Mutual information bound based on Noise-Contrastive Estimation

  Reference:
    Tschannen, M., Djolonga, J., Rubenstein, P.K., Gelly, S., Lucic, M., 2019.
      "On Mutual Information Maximization for Representation Learning".
      arXiv:1907.13625 [cs, stat].
    https://github.com/google-research/google-research/tree/master/mutual_information_representation_learning
  """


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
  pass


class MutualInfoVAE(BetaVAE):
  r""" Lambda is replaced as gamma in this implementation

  The algorithm of MI-VAE is as following:
  ```
  1. Compute q(z,c|x) and the KL-Divergence from the prior p(z).
  2. Generatea sample (z, c) from the approximate posterior q.
  3. Compute the conditional p(x|z) and incur the reconstruction loss.
  4. Resample (z_prime, c_prime) ~ p(c,z) from the prior.
  5. Recompute the conditional p(x|z_prime, c_prime) and generate a sample x_prime.
  6. Recompute the approximate posterior q(c|x_prime) and incur the loss for the MI lower bound.
  ```

  Reference:
    Ducau, F.N., Trénous, S. "Mutual Information in Variational Autoencoders".
      (2017) https://github.com/fducau/infoVAE.
    Chen, X., Chen, X., Duan, Y., et al. (2016) "InfoGAN: Interpretable
      Representation Learning by Information Maximizing Generative
      Adversarial Nets". URL : http://arxiv.org/ abs/1606.03657.
    Ducau, F.N. Code:  https://github.com/fducau/infoVAE
  """

  def __init__(self,
               beta=1.0,
               gamma=1.0,
               latents=RandomVariable(event_shape=5,
                                      posterior='diag',
                                      projection=True,
                                      name="Latent"),
               code=RandomVariable(event_shape=5,
                                   posterior='diag',
                                   projection=True,
                                   name='Code'),
               resample_zprime=False,
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
    self.resample_zprime = bool(resample_zprime)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    # don't take KL of qC_X
    llk, div = super()._elbo(X,
                             pX_Z,
                             qZ_X[:-1],
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training)
    # the latents, in the implementation, the author reuse z samples here,
    # but in the algorithm, z_prime is re-sampled from the prior.
    # But, reasonably, we want to hold z_prime fix to z, and c_prime is the
    # only change factor here.
    if not self.resample_zprime:
      z_prime = tf.concat([tf.convert_to_tensor(q) for q in qZ_X[:-1]], axis=-1)
      batch_shape = z_prime.shape[:-1]
    else:
      batch_shape = qZ_X[0].batch_shape
      z_prime = tf.concat(
          [q.KL_divergence.prior.sample(batch_shape) for q in qZ_X[:-1]],
          axis=-1)
    # mutual information code
    qC_X = qZ_X[-1]
    c_prime = qC_X.KL_divergence.prior.sample(batch_shape)
    if self.is_binary_code:
      c_prime = tf.clip_by_value(c_prime, 1e-8, 1. - 1e-8)
    # decoding
    samples = tf.concat([z_prime, c_prime], axis=-1)
    pX_Zprime = self.decode(samples, training=training)
    qC_Xprime = self.encode(pX_Zprime, training=training)[-1]
    # mutual information (we want to maximize this, hence, add it to the llk)
    mi = qC_Xprime.log_prob(c_prime)
    llk['mi'] = self.gamma * mi
    return llk, div


class FactorInfoVAE(BetaVAE):
  r""" This idea combining FactorVAE (Kim et al. 2018) and
  MutualInfoVAE (Ducau et al. 2017)
  # TODO

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
    Ducau, F.N., Trénous, S., 2017."Mutual Information in Variational
      Autoencoders". https://github.com/fducau/infoVAE.
  """

  def __init__(self, beta=1.0, gamma=1.0, **kwargs):
    super().__init__(beta=beta, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    # don't take KL of qC_X
    llk, div = super()._elbo(X,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training)
    z_prime = [permute_dims(q) for q in qZ_X]
    pX_Zprime = self.decode(z_prime, training=training)
    qZ_Xprime = self.encode(pX_Zprime, training=training)
    div['mmd'] = self.gamma * maximum_mean_discrepancy(
        qZ=qZ_Xprime,
        pZ=qZ_X[0].KL_divergence.prior,
        q_sample_shape=None,
        p_sample_shape=100)
    return llk, div
