from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python.distributions import (PowerSpherical,
                                                         SphericalUniform,
                                                         VonMisesFisher)
from tensorflow_probability.python.layers import DistributionLambda
from typing_extensions import Literal

from odin.backend.interpolation import Interpolation, linear
from odin.bay.layers.dense_distribution import DistributionDense
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import BetaVAE

__all__ = ['HypersphericalVAE', 'PowersphericalVAE']


class _von_mises_fisher:

  def __init__(self, event_size):
    self.event_size = int(event_size)

  def __call__(self, x):
    # use softplus1 for concentration to prevent collapse and instability with
    # small concentration
    # note in the paper:
    # z_var = tf.layers.dense(h1, units=1, activation=tf.nn.softplus) + 1
    return VonMisesFisher(
      mean_direction=tf.math.l2_normalize(x[..., :self.event_size], axis=-1),
      concentration=tf.nn.softplus(x[..., -1]),
    )


class _power_spherical:

  def __init__(self, event_size):
    self.event_size = int(event_size)

  def __call__(self, x):
    return PowerSpherical(
      mean_direction=tf.math.l2_normalize(x[..., :self.event_size], axis=-1),
      concentration=tf.nn.softplus(x[..., -1]),
    )


class HypersphericalVAE(BetaVAE):
  """Hyper-spherical VAE

  References
  -----------
  Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T. & Tomczak, J. M.
      Hyperspherical Variational Auto-Encoders. arXiv:1804.00891 [cs, stat] (2018).
  Davidson, T. R., Tomczak, J. M. & Gavves, E. Increasing Expressivity
      of a Hyperspherical VAE.
  Xu, J. & Durrett, G. Spherical Latent Spaces for Stable Variational
      Autoencoders. arXiv:1808.10805 [cs] (2018).
  De Cao, N. & Aziz, W. The Power Spherical distribution.
      arXiv:2006.04437 [cs, stat] (2020).
  """

  def __init__(
      self,
      latents: Union[RVconf, Layer] = RVconf(64, name="latents"),
      distribution: Literal[
        'powerspherical', 'vonmisesfisher'] = 'vonmisesfisher',
      prior: Union[
        None, SphericalUniform, VonMisesFisher, PowerSpherical] = None,
      beta: Union[float, Interpolation] = linear(vmin=1e-6,
                                                 vmax=1.,
                                                 steps=2000,
                                                 delay_in=0),
      **kwargs):
    event_shape = latents.event_shape
    event_size = int(np.prod(event_shape))
    distribution = str(distribution).lower()
    assert distribution in ('powerspherical', 'vonmisesfisher'), \
      ('Support PowerSpherical or VonMisesFisher distribution, '
       f'but given: {distribution}')
    if distribution == 'powerspherical':
      fn_distribution = _power_spherical(event_size)
      default_prior = SphericalUniform(dimension=event_size)
    else:
      fn_distribution = _von_mises_fisher(event_size)
      default_prior = VonMisesFisher(0, 10)
    if prior is None:
      prior = default_prior
    latents = DistributionDense(
      event_shape,
      posterior=DistributionLambda(make_distribution_fn=fn_distribution),
      prior=prior,
      units=event_size + 1,
      name=latents.name)
    super().__init__(latents=latents,
                     analytic=True,
                     beta=beta,
                     **kwargs)


class PowersphericalVAE(HypersphericalVAE):

  def __init__(self, **kwargs):
    kwargs.pop('distribution')
    super().__init__(distribution='powerspherical', **kwargs)


class poincareVAE(BetaVAE):
  ...
