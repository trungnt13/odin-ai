import inspect
from functools import partial
from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TensorTypes
from odin.bay.vi.losses import get_divergence, maximum_mean_discrepancy
from odin.bay.vi.utils import permute_dims
from odin.utils import as_tuple
from tensorflow import Tensor
from tensorflow_probability.python.distributions import (FULLY_REPARAMETERIZED,
                                                         NOT_REPARAMETERIZED,
                                                         Distribution,
                                                         OneHotCategorical)
from typing_extensions import Literal


# ===========================================================================
# Helpers
# ===========================================================================
def _clip_binary(x, eps=1e-7):
  # this is ad-hoc value, tested 1e-8 but return NaN for RelaxedSigmoid
  # all the time
  return tf.clip_by_value(x, eps, 1. - eps)


# ===========================================================================
# InfoVAE
# ===========================================================================
class infoVAE(betaVAE):
  r""" For MNIST, the authors used scaling coefficient `lambda=1000`,
  and information preference `alpha=0`.

  Increase `np` (number of prior samples) in `divergence_kw` to reduce the
  variance of MMD estimation.

  Arguments:
    alpha : float
      Equal to `1 - beta`. Higher value of alpha places lower weight
      on the KL-divergence
    lamda : float
      This is the value of lambda in the paper.
      Higher value of lambda place more weight on the Info-divergence (i.e. MMD)
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
      lamda: float = 100.0,
      divergence: Callable[[Distribution, Distribution],
                           Tensor] = partial(maximum_mean_discrepancy,
                                             kernel='gaussian',
                                             q_sample_shape=None,
                                             p_sample_shape=100),
      name='InfoVAE',
      **kwargs,
  ):
    super().__init__(beta=1 - alpha, name=name, **kwargs)
    self.lamda = tf.convert_to_tensor(lamda, dtype=self.dtype, name='lambda')
    # select right divergence
    assert callable(divergence), \
      f"divergence must be callable, but given: {type(divergence)}"
    self.divergence = divergence

  @property
  def alpha(self):
    return 1 - self.beta

  @alpha.setter
  def alpha(self, alpha):
    self.beta = 1 - alpha

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    # repeat for each latent
    for z, qz in zip(as_tuple(self.latents), as_tuple(qz_x)):
      # div(qZ||pZ)
      info_div = (self.lamda - self.beta) * self.divergence(
          qz, qz.KL_divergence.prior)
      kl[f'div_{z.name}'] = info_div
    return llk, kl


# ===========================================================================
# Mutual Information VAE
# ===========================================================================
class miVAE(betaVAE):
  r""" Mutual-information VAE

  The algorithm of MI-VAE is as following:
  ```
  1. Compute q(z,c|x) and the KL-Divergence from the prior p(z).
  2. Generatea sample (z, c) from the approximate posterior q.
  3. Compute the conditional p(x|z) and incur the reconstruction loss.
  ---
  4. Resample (z_prime, c_prime) ~ p(c,z) from the prior.
  5. Recompute the conditional p(x|z_prime, c_prime) and generate a sample x_prime.
  6. Recompute the approximate posterior q(c|x_prime)
  7. Incur the loss for the MI lower bound q(c|x_prime).log_prob(c_prime).
  ```

  Parameters
  ----------
  minimize_kl_codes : a Boolean (default: True).
    If False, only maximize the mutual information of the factors code
    `q(c|X)` and the input `p(X|z, c)`, this is the original configuration
    in the paper.
    If True, encourage mutual code to be factorized as well by minimizing
    the KL divergence to the multivariate diagonal Gaussian piror.

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
               mi_coef: float = 1.0,
               kl_codes_coef: float = 0.,
               latents: RVmeta = RVmeta(32,
                                        'mvndiag',
                                        projection=True,
                                        name='Latents'),
               mutual_codes: RVmeta = RVmeta(32,
                                             'mvndiag',
                                             projection=True,
                                             name='Codes'),
               steps_without_mi: int = 100,
               **kwargs):
    self.is_binary_code = mutual_codes.is_binary
    super().__init__(beta=beta, latents=latents, **kwargs)
    self.mutual_codes = mutual_codes.create_posterior()
    self.mi_coef = float(mi_coef)
    self.kl_codes_coef = float(kl_codes_coef)
    self.steps_without_mi = int(steps_without_mi)

  def sample_prior(self,
                   sample_shape: Union[int, List[int]] = (),
                   seed: int = 1) -> Tensor:
    r""" Sampling from prior distribution """
    z1 = super().sample_prior(sample_shape=sample_shape, seed=seed)
    z2 = self.mutual_codes.prior.sample(sample_shape, seed=seed)
    return (z1, z2)

  def encode(self, inputs, **kwargs):
    h_e = self.encoder(inputs, **kwargs)
    # create the latents distribution
    qz_x = self.latents(h_e, **kwargs)
    qc_x = self.mutual_codes(h_e, **kwargs)
    # need to keep the keras mask
    mask = kwargs.get('mask', None)
    qz_x._keras_mask = mask
    qc_x._keras_mask = mask
    return (qz_x, qc_x)

  def decode(self, latents, **kwargs):
    latents = tf.concat(latents, axis=-1)
    return super().decode(latents, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    # NOTE: the original implementation does not take KL(qC_X||pC),
    # only maximize the mutual information of q(c|X)
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, (qz_x, qc_x) = self.last_outputs
    ## This approach is not working!
    # z_prime = tf.stop_gradient(tf.convert_to_tensor(qz_x))
    # batch_shape = z_prime.shape[:-1]
    # c_prime = qc_x.KL_divergence.prior.sample(batch_shape)
    ##
    batch_shape = px_z.batch_shape
    z_prime = qz_x.KL_divergence.prior.sample(batch_shape)
    c_prime = qc_x.KL_divergence.prior.sample(batch_shape)
    ## clip to prevent underflow for relaxed-bernoulli
    if self.is_binary_code:
      c_prime = _clip_binary(c_prime)
    ## decoding
    px = self.decode([z_prime, c_prime], training=training)
    if px.reparameterization_type == NOT_REPARAMETERIZED:
      x = px.mean()
    else:
      x = tf.convert_to_tensor(px)
    qz_xprime, qc_xprime = self.encode(x, training=training)
    #' mutual information (we want to maximize this, hence, add it to the llk)
    mi_c = qc_xprime.log_prob(c_prime)
    llk['mi_codes'] = tf.cond(self.step > self.steps_without_mi,
                              true_fn=lambda: self.mi_coef * mi_c,
                              false_fn=lambda: 0.)
    ## this value is just for monitoring
    mi_z = qz_xprime.log_prob(z_prime)
    llk['mi_latents'] = tf.stop_gradient(mi_z)
    ## factorizing the mutual codes if required
    if hasattr(qc_x, 'KL_divergence'):
      kl_c = qc_x.KL_divergence()
    else:
      kl_c = 0.
    if self.kl_codes_coef == 0.:
      kl_c = tf.stop_gradient(kl_c)
    else:
      kl_c = self.kl_codes_coef * kl_c
    kl['kl_codes'] = kl_c
    return llk, kl


class SemiInfoVAE(miVAE):
  r""" This idea combining factorVAE (Kim et al. 2018) and
  miVAE (Ducau et al. 2017)

  # TODO
  """
  ...


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
