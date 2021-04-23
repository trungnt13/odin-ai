from functools import partial
from typing import Callable, List, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow_probability.python.distributions import (NOT_REPARAMETERIZED,
                                                         Distribution)

from odin.backend.interpolation import linear
from odin.backend.types_helpers import Coefficient
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.losses import maximum_mean_discrepancy
from odin.utils import as_tuple


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
class InfoVAE(BetaVAE):
  """ For MNIST, the authors used scaling coefficient `lambda=1000`,
  and information preference `alpha=0`.

  Increase `np` (number of prior samples) in `divergence_kw` to reduce the
  variance of MMD estimation.

  Parameters
  ----------
  alpha : float
    Equal to `1 - beta`. Higher value of alpha places lower weight
    on the KL-divergence
  lamda : float
    This is the value of lambda in the paper.
    Higher value of lambda place more weight on the Info-divergence
    (i.e. MMD)
  divergence : a Callable.
    Divergences families, for now only support 'mmd'
    i.e. maximum-mean discrepancy.

  References
  ----------
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
    kwargs.pop('beta')
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

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    # repeat for each latent
    for layer, qz in zip(as_tuple(self.latents), as_tuple(qz_x)):
      # div(qZ||pZ)
      info_div = self.divergence(qz, qz.KL_divergence.prior)
      kl[f'div_{layer.name}'] = (self.lamda - self.beta) * info_div
    return llk, kl


# ===========================================================================
# Mutual Information VAE
# ===========================================================================
class MIVAE(BetaVAE):
  """ Mutual-information VAE

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

  def __init__(
      self,
      mi_coef: Coefficient = 0.2,
      latents: RVconf = RVconf(32, 'mvndiag', projection=True, name='latents'),
      mutual_codes: RVconf = RVconf(10,
                                    'mvndiag',
                                    projection=True,
                                    name='codes'),
      steps_without_mi: int = 100,
      beta: Coefficient = linear(vmin=1e-6, vmax=1., steps=2000),
      beta_codes: Coefficient = 0.,
      name: str = 'MutualInfoVAE',
      **kwargs,
  ):
    super().__init__(beta=beta, latents=latents, name=name, **kwargs)
    self.is_binary_code = mutual_codes.is_binary
    if isinstance(mutual_codes, RVconf):
      mutual_codes = mutual_codes.create_posterior()
    self.mutual_codes = mutual_codes
    self._mi_coef = mi_coef
    self._beta_codes = beta_codes
    self.steps_without_mi = int(steps_without_mi)

  @classmethod
  def is_hierarchical(self) -> bool:
    return True

  @property
  def beta_codes(self) -> tf.Tensor:
    if callable(self._beta_codes):
      return self._beta_codes(self.step)
    return tf.constant(self._beta_codes, dtype=self.dtype)

  @property
  def mi_coef(self) -> tf.Tensor:
    if callable(self._mi_coef):
      return self._mi_coef(self.step)
    return tf.constant(self._mi_coef, dtype=self.dtype)

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
    ## factorizing the mutual codes if required
    kl_c = qc_x.KL_divergence(free_bits=self.free_bits)
    kl[f'kl_{self.mutual_codes.name}'] = tf.cond(
      self.beta_codes > 1e-8,  # for numerical stability
      true_fn=lambda: self.beta_codes * kl_c,
      false_fn=lambda: tf.stop_gradient(kl_c),
    )
    ## This approach is not working!
    # z_prime = tf.stop_gradient(tf.convert_to_tensor(qz_x))
    # batch_shape = z_prime.shape[:-1]
    # c_prime = qc_x.KL_divergence.prior.sample(batch_shape)
    ## sampling for maximizing I(X;Z)
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
    ## mutual information (we want to maximize this, hence, add it to the llk)
    llk['mi_codes'] = self.mi_coef * tf.cond(
      self.step > self.steps_without_mi,
      true_fn=lambda: qc_xprime.log_prob(c_prime),
      false_fn=lambda: 0.)
    ## this value is just for monitoring
    mi_z = tf.stop_gradient(tf.reduce_mean(qz_xprime.log_prob(z_prime)))
    llk['mi_latents'] = tf.cond(
      tf.logical_or(tf.math.is_nan(mi_z), tf.math.is_inf(mi_z)),
      true_fn=lambda: 0.,
      false_fn=lambda: mi_z,
    )
    return llk, kl

# class InfoNCEVAE(betaVAE):
#   r""" Mutual information bound based on Noise-Contrastive Estimation

#   Reference:
#     Tschannen, M., Djolonga, J., Rubenstein, P.K., Gelly, S., Lucic, M., 2019.
#       "On Mutual Information Maximization for Representation Learning".
#       arXiv:1907.13625 [cs, stat].
#     https://github.com/google-research/google-research/tree/master/mutual_information_representation_learning
#   """

# class IFVAE(betaVAE):
#   r""" Adversarial information factorized VAE

#   Reference:
#     Creswell, A., Mohamied, Y., Sengupta, B., Bharath, A.A., 2018.
#       "Adversarial Information Factorization". arXiv:1711.05175 [cs].
#   """

# class InfoMaxVAE(betaVAE):
#   r"""
#   Reference:
#     Rezaabad, A.L., Vishwanath, S., 2020. "Learning Representations by
#       Maximizing Mutual Information in Variational Autoencoders".
#       arXiv:1912.13361 [cs, stat].
#     Hjelm, R.D., Fedorov, A., et al. 2019. "Learning Deep Representations by
#       Mutual Information Estimation and Maximization". ICLR'19.
#   """
