import tensorflow as tf

from odin.backend import interpolation as interp
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.losses import total_correlation


class BetaVAE(VariationalAutoencoder):
  r""" Implementation of beta-VAE
  Arguments:
    beta : a Scalar. A regularizer weight indicate the capacity of the latent.

  Reference:
    Higgins, I., Matthey, L., Pal, A., et al. "beta-VAE: Learning Basic
      Visual Concepts with a Constrained Variational Framework".
      ICLR'17
  """

  def __init__(self, beta=10.0, **kwargs):
    super().__init__(**kwargs)
    self.beta = beta

  @property
  def beta(self):
    if isinstance(self._beta, interp.Interpolation):
      return self._beta(self.step)
    return self._beta

  @beta.setter
  def beta(self, b):
    if isinstance(b, interp.Interpolation):
      self._beta = b
    else:
      self._beta = tf.convert_to_tensor(b, dtype=self.dtype, name='beta')

  def _elbo(self, inputs, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training, **kwargs):
    llk, div = super()._elbo(inputs,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training,
                             **kwargs)
    div = {key: self.beta * val for key, val in div.items()}
    return llk, div


class BetaTCVAE(BetaVAE):
  r""" Extend the beta-VAE with total correlation loss added.

  Based on Equation (4) with alpha = gamma = 1
  If alpha = gamma = 1, Eq. 4 can be written as
    `ELBO = LLK - (KL + (beta - 1) * TC)`.

  Reference:
    Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. "Isolating Sources
      of Disentanglement in Variational Autoencoders".
      arXiv:1802.04942 [cs, stat].
  """

  def _elbo(self, inputs, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training, **kwargs):
    llk, div = super()._elbo(inputs,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training,
                             **kwargs)
    for name, q in zip(self.latent_names, qZ_X):
      tc = total_correlation(q.sample(), q)
      div['tc_%s' % name] = (self.beta - 1.) * tc
    return llk, div


class AnnealedVAE(VariationalAutoencoder):
  r"""Creates an AnnealedVAE model.

  Implementing Eq. 8 of (Burgess et al. 2018)

  Arguments:
    gamma: Hyperparameter for the regularizer.
    c_max: a Scalar. Maximum capacity of the bottleneck.
      is gradually increased from zero to a value large enough to produce
      good quality reconstructions
    iter_max: an Integer. Number of iteration until reach the maximum
      capacity (start from 0).
    interpolation : a String. Type of interpolation for increasing capacity.

  Example:
    vae = AnnealedVAE()
    elbo = vae.elbo(x, px, qz, n_iter=1)

  Reference:
    Burgess, C.P., Higgins, I., et al. 2018. "Understanding disentangling in
      beta-VAE". arXiv:1804.03599 [cs, stat].
  """

  def __init__(self,
               gamma=1.0,
               c_min=0.,
               c_max=25.,
               iter_max=1000,
               interpolation='linear',
               **kwargs):
    super().__init__(**kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.interpolation = interp.get(str(interpolation))(
        vmin=tf.constant(c_min, self.dtype),
        vmax=tf.constant(c_max, self.dtype),
        norm=int(iter_max))

  def _elbo(self, inputs, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training, **kwargs):
    llk, div = super()._elbo(inputs,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training,
                             **kwargs)
    # step : training step, updated when call `.train_steps()`
    c = self.interpolation(self.step)
    div = {key: self.gamma * tf.math.abs(val - c) for key, val in div.items()}
    return llk, div


# class CyclicalAnnealingVAE(BetaVAE):
#   r"""
#   Reference:
#     Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., Carin, L., 2019.
#       "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
#       Vanishing". arXiv:1903.10145 [cs, stat].
#   """
#   pass
