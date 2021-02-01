from functools import partial
from typing import Callable, List, Optional, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow_probability.python.distributions import (NOT_REPARAMETERIZED,
                                                         Distribution)

from odin.backend.interpolation import Interpolation, circle, linear
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.bay.vi.losses import maximum_mean_discrepancy
from odin.bay.vi.utils import prepare_ssl_inputs
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
class infoVAE(betaVAE):
  """ For MNIST, the authors used scaling coefficient `lambda=1000`,
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
                                        name='latents'),
               mutual_codes: Optional[RVmeta] = None,
               steps_without_mi: int = 100,
               name='miVAE',
               **kwargs):
    super().__init__(beta=beta, latents=latents, name=name, **kwargs)
    if mutual_codes is None:
      zdim = sum(sum(q.event_shape) for q in as_tuple(self.latents))
      mutual_codes = RVmeta(zdim, 'mvndiag', projection=True, name='Codes')
    self.is_binary_code = mutual_codes.is_binary
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
    if training:
      llk['mi_codes'] = tf.cond(self.step > self.steps_without_mi,
                                true_fn=lambda: self.mi_coef * mi_c,
                                false_fn=lambda: 0.)
    else:
      llk['mi_codes'] = self.mi_coef * mi_c
    ## this value is just for monitoring
    mi_z = tf.reduce_mean(qz_xprime.log_prob(z_prime))
    mi_z = tf.cond(tf.math.is_nan(mi_z),
                   true_fn=lambda: 0.,
                   false_fn=lambda: tf.clip_by_value(mi_z, -1e8, 1e8))
    llk['mi_latents'] = tf.stop_gradient(mi_z)
    ## factorizing the mutual codes if required
    if hasattr(qc_x, 'KL_divergence'):
      kl_c = qc_x.KL_divergence(free_bits=self.free_bits)
    else:
      kl_c = 0.
    if self.kl_codes_coef == 0.:
      kl_c = tf.stop_gradient(kl_c)
    if training:
      kl_c = self.kl_codes_coef * kl_c
    kl['kl_codes'] = kl_c
    return llk, kl


class semafoVAE(betaVAE):
  """A semaphore is a variable or abstract data type used to control access to
  a common resource by multiple processes and avoid critical section problems in
  a concurrent system

  SemafoVAE  [Callback#50001]:
  llk_x:-73.33171081542969
  llk_y:-0.9238954782485962
  acc_y:0.7268000245094299

  Without autoregressive
  llk_x:-72.9976577758789
  llk_y:-0.7141319513320923
  acc_y:0.8095999956130981
  """

  def __init__(
      self,
      labels: RVmeta = RVmeta(10, 'onehot', projection=True, name="digits"),
      alpha: float = 10.0,
      mi_coef: Union[float, Interpolation] = circle(vmin=0.,
                                                    vmax=0.1,
                                                    length=2000,
                                                    delay_in=100,
                                                    delay_out=100,
                                                    cyclical=True),
      beta: Union[float, Interpolation] = linear(vmin=1e-6,
                                                 vmax=1.,
                                                 length=2000,
                                                 delay_in=0),
      name='SemafoVAE',
      **kwargs,
  ):
    super().__init__(beta=beta, name=name, **kwargs)
    self.labels = _parse_layers(labels)
    self._mi_coef = mi_coef
    self.alpha = alpha

  def build(self, input_shape):
    return super().build(input_shape)

  @property
  def mi_coef(self):
    if isinstance(self._mi_coef, Interpolation):
      return self._mi_coef(self.step)
    return self._mi_coef

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    # don't condition on the labels, only accept inputs
    X = X[0]
    qz_x = super().encode(X, training=training, mask=None, **kwargs)
    qy_zx = self.labels(tf.convert_to_tensor(qz_x),
                        training=training,
                        mask=mask)
    return (qz_x, qy_zx)

  def decode(self, latents, training=None, mask=None, **kwargs):
    if isinstance(latents, (tuple, list)):
      latents = latents[0]
    return super().decode(latents, training, mask, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    px_z, (qz_x, py_zx) = self.last_outputs
    ## supervised loss
    if len(y) > 0:
      llk_y = py_zx.log_prob(y[0])
      if mask is not None:
        llk_y = tf.cond(
            tf.reduce_all(tf.logical_not(mask)),
            lambda: 0.,
            lambda: tf.transpose(
                tf.boolean_mask(tf.transpose(llk_y), mask, axis=0)),
        )
      llk_y = tf.reduce_mean(self.alpha * llk_y)
      llk_y = tf.cond(tf.abs(llk_y) < 1e-8,
                      true_fn=lambda: tf.stop_gradient(llk_y),
                      false_fn=lambda: llk_y)
      llk[f"llk_{self.labels.name}"] = llk_y
    ## sample the prior
    batch_shape = qz_x.batch_shape
    z_prime = qz_x.KL_divergence.prior.sample(batch_shape)
    ## decoding
    px = self.decode(z_prime, training=training)
    if px.reparameterization_type == NOT_REPARAMETERIZED:
      x = px.mean()
    else:
      x = tf.convert_to_tensor(px)
    # x = tf.stop_gradient(x) # should not stop gradient here, generator need to be updated
    qz_xprime, qy_zxprime = self.encode(x, training=training)
    #' mutual information (we want to maximize this, hence, add it to the llk)
    y = tf.convert_to_tensor(py_zx)
    # only calculate MI for unsupervised data
    mi_y = tf.reduce_mean(
        tf.boolean_mask(
            py_zx.log_prob(y) - qy_zxprime.log_prob(y), tf.logical_not(mask)))
    if training:
      llk[f'mi_{self.labels.name}'] = self.mi_coef * mi_y
    else:
      llk[f'mi_{self.labels.name}'] = mi_y
    ## this value is just for monitoring
    mi_z = tf.reduce_mean(qz_xprime.log_prob(z_prime))
    mi_z = tf.cond(tf.math.is_nan(mi_z),
                   true_fn=lambda: 0.,
                   false_fn=lambda: tf.clip_by_value(mi_z, -1e8, 1e8))
    llk[f'mi_{self.latents.name}'] = tf.stop_gradient(mi_z)
    return llk, kl


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


class InfoMaxVAE(betaVAE):
  r"""
  Reference:
    Rezaabad, A.L., Vishwanath, S., 2020. "Learning Representations by
      Maximizing Mutual Information in Variational Autoencoders".
      arXiv:1912.13361 [cs, stat].
    Hjelm, R.D., Fedorov, A., et al. 2019. "Learning Deep Representations by
      Mutual Information Estimation and Maximization". ICLR'19.
  """
