import tensorflow as tf
from typing import Union
from odin.backend import interpolation as interp
from odin.backend.interpolation import Interpolation, linear
from odin.bay.vi.autoencoder.variational_autoencoder import VariationalAutoencoder
from odin.bay.vi.losses import total_correlation
from odin.utils import as_tuple


class betaVAE(VariationalAutoencoder):
  """ Implementation of beta-VAE

  Parameters
  ----------
  beta : a Scalar.
      A regularizer weight indicate the capacity of the latent.

  Reference
  -----------
  Higgins, I., Matthey, L., Pal, A., et al. "beta-VAE: Learning Basic
      Visual Concepts with a Constrained Variational Framework".
      ICLR'17
  """

  def __init__(self,
               beta: Union[float, Interpolation] = 1.0,
               name='BetaVAE',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._beta = beta

  @property
  def beta(self) -> tf.Tensor:
    if callable(self._beta):
      return self._beta(self.step)
    return tf.constant(self._beta, dtype=self.dtype)

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    kl = {key: self.beta * val for key, val in kl.items()}
    return llk, kl


class beta10VAE(betaVAE):

  def __init__(self, name='Beta10VAE', **kwargs):
    kwargs.pop('beta', None)
    super().__init__(beta=10.0, name=name, **kwargs)


class annealingVAE(betaVAE):
  """ KL-annealing VAE, cyclical annealing could be achieved by setting
  `cyclical=True` when creating `Interpolation`

  References
  ----------
  Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., Carin, L., 2019.
      "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
      Vanishing". arXiv:1903.10145 [cs, stat].
  Maaløe, L., Sønderby, C.K., Sønderby, S.K., Winther, O., 2016. Auxiliary
      Deep Generative Models. arXiv:1602.05473 [cs, stat].
  Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K., Winther, O., 2016.
      Ladder variational autoencodersself. Advances in Neural Information
      Processing Systems. Curran Associates, Inc., pp. 3738–3746.
  """

  def __init__(
      self,
      beta: Union[float, Interpolation] = linear(vmin=1e-6,
                                                 vmax=1.0,
                                                 length=2000,
                                                 delay_in=0),
      name: str = 'AnnealingVAE',
      **kwargs,
  ):
    super().__init__(beta=beta, name=name, **kwargs)


class betatcVAE(betaVAE):
  r""" Extend the beta-VAE with total correlation loss added.

  Based on Equation (4) with alpha = gamma = 1
  If alpha = gamma = 1, Eq. 4 can be written as
    `ELBO = LLK - (KL + (beta - 1) * TC)`.

  Reference:
    Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. "Isolating Sources
      of Disentanglement in Variational Autoencoders".
      arXiv:1802.04942 [cs, stat].
  """

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    for z, qz in zip(as_tuple(self.latents), as_tuple(qz_x)):
      tc = total_correlation(tf.convert_to_tensor(qz), qz)
      kl[f'tc_{z.name}'] = (self.beta - 1.) * tc
    return llk, kl


class capacitateVAE(VariationalAutoencoder):
  r"""Creates an annealedVAE model.

  Implementing Eq. 8 of (Burgess et al. 2018)

  Parameters
  ----------
  gamma: Hyperparameter for the regularizer.
  c_max: a Scalar. Maximum capacity of the bottleneck.
    is gradually increased from zero to a value large enough to produce
    good quality reconstructions
  iter_max: an Integer. Number of iteration until reach the maximum
    capacity (start from 0).
  interpolation : a String. Type of interpolation for increasing capacity.

  Example
  -------
  vae = annealedVAE()
  elbo = vae.elbo(x, px, qz, n_iter=1)

  Reference
  ---------
  Burgess, C.P., Higgins, I., et al. 2018. "Understanding disentangling in
    beta-VAE". arXiv:1804.03599 [cs, stat].
  """

  def __init__(self,
               gamma: float = 1.0,
               c_min: float = 0.,
               c_max: float = 25.,
               iter_max: int = 1000,
               interpolation: str = 'linear',
               **kwargs):
    super().__init__(**kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.interpolation = interp.get(str(interpolation))(
        vmin=tf.constant(c_min, self.dtype),
        vmax=tf.constant(c_max, self.dtype),
        norm=int(iter_max))

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    # step : training step, updated when call `.train_steps()`
    c = self.interpolation(self.step)
    kl = {key: self.gamma * tf.math.abs(val - c) for key, val in kl.items()}
    return llk, kl
