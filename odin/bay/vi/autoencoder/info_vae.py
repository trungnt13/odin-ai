import inspect
from functools import partial
from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TensorTypes
from odin.bay.vi.losses import maximum_mean_discrepancy
from odin.bay.vi.utils import permute_dims
from odin.utils import as_tuple
from tensorflow import Tensor
from tensorflow_probability.python.distributions import (Distribution,
                                                         OneHotCategorical)
from typing_extensions import Literal


class infoVAE(betaVAE):
  r""" For MNIST, the authors used scaling coefficient `lambda(gamma)=1000`,
  and information preference `alpha=0`.

  Increase `np` (number of prior samples) in `divergence_kw` to reduce the
  variance of MMD estimation.

  Arguments:
    alpha : a Scalar. Equal to `1 - beta`
      Higher value of alpha places lower weight on the KL-divergence
    gamma : a Scalar. This is the value of lambda in the paper
      Higher value of gamma place more weight on the Info-divergence (i.e. MMD)
    divergence : a Callable.
      Divergences families, for now only support 'mmd'
      i.e. maximum-mean discrepancy.

  Reference:
    Zhao, S., Song, J., Ermon, S., et al. "infoVAE: Balancing Learning and
      Inference in Variational Autoencoders".
    Shengjia Zhao. "A Tutorial on Information Maximizing Variational
      Autoencoders (infoVAE)".
      https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders
  """

  def __init__(
      self,
      alpha: float = 0.0,
      gamma: float = 100.0,
      divergence: Callable[[Distribution, Distribution],
                           Tensor] = partial(maximum_mean_discrepancy,
                                             kernel='gaussian',
                                             q_sample_shape=None,
                                             p_sample_shape=100),
      **kwargs,
  ):
    super().__init__(beta=1 - alpha, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    # select right divergence
    assert callable(divergence), \
      f"divergence must be callable, but given: {type(divergence)}"
    self.divergence = divergence

  @property
  def alpha(self):
    return 1 - self.beta

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    # repeat for each latent
    for z, qz in zip(as_tuple(self.latents), as_tuple(qz_x)):
      # div(qZ||pZ)
      info_div = (self.gamma - self.beta) * self.divergence(
          qz, qz.KL_divergence.prior)
      kl[f'div_{z.name}'] = info_div
    return llk, kl


from numbers import Number

import tensorflow as tf
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.losses import get_divergence, maximum_mean_discrepancy


def _clip_binary(x, eps=1e-7):
  # this is ad-hoc value, tested 1e-8 but return NaN for RelaxedSigmoid
  # all the time
  return tf.clip_by_value(x, eps, 1. - eps)


class miVAE(betaVAE):
  r""" Mutual-information VAE

  The algorithm of MI-VAE is as following:
  ```
  1. Compute q(z,c|x) and the KL-Divergence from the prior p(z).
  2. Generatea sample (z, c) from the approximate posterior q.
  3. Compute the conditional p(x|z) and incur the reconstruction loss.
  4. Resample (z_prime, c_prime) ~ p(c,z) from the prior.
  5. Recompute the conditional p(x|z_prime, c_prime) and generate a sample x_prime.
  6. Recompute the approximate posterior q(c|x_prime) and incur the loss for the MI lower bound.
  ```

  Parameters
  ----------
  resample_zprime : a Boolean. if True, use samples from q(z|x) for z_prime
    instead of sampling z_prime from prior.
  kl_factors : a Boolean (default: True).
    If False, only maximize the mutual information of the factors code
    `q(c|X)` and the input `p(X|z, c)`, this is the original configuration
    in the paper.
    If True, encourage factorized code by pushing the KL divergence to the
    prior (multivariate diagonal normal).

  Note
  -----
  Lambda is replaced as gamma in this implementation


  References
  ----------
  Ducau, F.N., Trénous, S. "Mutual Information in Variational Autoencoders".
    (2017) https://github.com/fducau/infoVAE.
  Chen, X., Chen, X., Duan, Y., et al. (2016) "InfoGAN: Interpretable
    Representation Learning by Information Maximizing Generative
    Adversarial Nets". URL : http://arxiv.org/ abs/1606.03657.
  Ducau, F.N. Code:  https://github.com/fducau/infoVAE
  """

  def __init__(self,
               beta: float = 1.0,
               gamma: float = 1.0,
               latents: RVmeta = RVmeta(5,
                                        'mvndiag',
                                        projection=True,
                                        name="Latents"),
               factors: RVmeta = RVmeta(5,
                                        'mvndiag',
                                        projection=True,
                                        name='Factors'),
               resample_zprime: bool = False,
               kl_factors: bool = True,
               **kwargs):
    self.is_binary_factors = factors.is_binary
    super().__init__(beta=beta, latents=latents, **kwargs)
    self.factors = self.latents[-1]
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.resample_zprime = bool(resample_zprime)
    self.kl_factors = bool(kl_factors)

  def decode(self, latents, training=None, mask=None, **kwargs):
    if isinstance(latents, (tuple, list)) and len(latents) > 1:
      latents = tf.concat(latents, axis=-1)
    return super().decode(latents, training=training, mask=mask, **kwargs)

  def elbo_components(self,
                      inputs,
                      training=None,
                      pX_Z=None,
                      qZ_X=None,
                      mask=None,
                      **kwargs):
    # NOTE: the original implementation does not take KL(qC_X||pC),
    # only maximize the mutual information of q(c|X)
    pX_Z, qZ_X = self.call(inputs,
                           training=training,
                           pX_Z=pX_Z,
                           qZ_X=qZ_X,
                           mask=mask,
                           **kwargs)
    pX_Z = tf.nest.flatten(pX_Z)
    qZ_X = tf.nest.flatten(qZ_X)
    llk, kl = super().elbo_components(
        inputs,
        pX_Z=pX_Z,
        qZ_X=qZ_X[:-1] if not self.kl_factors else qZ_X,
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
    if self.is_binary_factors:
      c_prime = _clip_binary(c_prime)
    # decoding
    samples = tf.concat([z_prime, c_prime], axis=-1)
    pX_Zprime = self.decode(samples, training=training)
    qC_Xprime = self.encode(pX_Zprime, training=training)[-1]
    # mutual information (we want to maximize this, hence, add it to the llk)
    mi = qC_Xprime.log_prob(c_prime)
    llk['mi'] = self.gamma * mi
    return llk, kl


class SemiInfoVAE(miVAE):
  r""" This idea combining factorVAE (Kim et al. 2018) and
  miVAE (Ducau et al. 2017)

  # TODO
  """

  def __init__(self, alpha=1., **kwargs):
    super().__init__(**kwargs)
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name="alpha")

  @property
  def is_semi_supervised(self):
    return True

  def encode(self, inputs, training=None, mask=None, sample_shape=(), **kwargs):
    inputs = tf.nest.flatten(inputs)
    if len(inputs) > len(self.observation):
      inputs = inputs[:len(self.observation)]
    return super().encode(inputs[0] if len(inputs) == 1 else inputs,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)

  def _elbo(self, inputs, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training, **kwargs):
    y = None
    if len(inputs) > len(pX_Z):
      y = inputs[-1]
    # don't take KL of qC_X
    llk, div = super(miVAE,
                     self)._elbo(inputs,
                                 pX_Z,
                                 qZ_X[:-1] if not self.kl_factors else qZ_X,
                                 analytic=analytic,
                                 reverse=reverse,
                                 sample_shape=sample_shape,
                                 mask=mask,
                                 training=training,
                                 **kwargs)
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
    if self.is_binary_factors:
      c_prime = _clip_binary(c_prime)
    # decoding
    samples = tf.concat([z_prime, c_prime], axis=-1)
    pX_Zprime = self.decode(samples, training=training)
    qC_Xprime = self.encode(pX_Zprime, training=training)[-1]
    ## mutual information (we want to maximize this, hence, add it to the llk)
    if y is not None:  # label is provided
      # clip the value for RelaxedSigmoid distribution otherwise NaN
      if self.is_binary_factors:
        y = _clip_binary(y)
      ss = qC_Xprime.log_prob(y)
      if mask is not None:
        mi = qC_Xprime.log_prob(c_prime)
        mi = tf.where(tf.reshape(mask, (-1,)), self.alpha * ss, self.gamma * mi)
      else:
        mi = self.alpha * ss
    else:  # no label just use the sampled code
      mi = self.gamma * qC_Xprime.log_prob(c_prime)
    llk['mi'] = mi
    return llk, div


class InfoNCEVAE(betaVAE):
  r""" Mutual information bound based on Noise-Contrastive Estimation

  Reference:
    Tschannen, M., Djolonga, J., Rubenstein, P.K., Gelly, S., Lucic, M., 2019.
      "On Mutual Information Maximization for Representation Learning".
      arXiv:1907.13625 [cs, stat].
    https://github.com/google-research/google-research/tree/master/mutual_information_representation_learning
  """


class IFVAE(betaVAE):
  r""" Adversarial information factorized VAE

  Reference:
    Creswell, A., Mohamied, Y., Sengupta, B., Bharath, A.A., 2018.
      "Adversarial Information Factorization". arXiv:1711.05175 [cs].
  """
  pass


class InfoMaxVAE(betaVAE):
  r"""
  Reference:
    Rezaabad, A.L., Vishwanath, S., 2020. "Learning Representations by
      Maximizing Mutual Information in Variational Autoencoders".
      arXiv:1912.13361 [cs, stat].
    Hjelm, R.D., Fedorov, A., et al. 2019. "Learning Deep Representations by
      Mutual Information Estimation and Maximization". ICLR'19.
  """
  pass
