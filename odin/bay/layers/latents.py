from __future__ import absolute_import, division, print_function

from typing import Type

import tensorflow as tf
from tensorflow_probability.python.distributions import (Independent, LogNormal,
                                                         MultivariateNormalDiag,
                                                         Normal)

from odin.bay.layers.continuous import (LogNormalLayer, MultivariateNormalLayer,
                                        NormalLayer)
from odin.bay.layers.dense_distribution import (DistributionDense,
                                                MixtureDensityNetwork)

__all__ = [
    'MultivariateNormalDiagLatent',
    'IndependentNormalLatent',
    'MixtureMultivariateNormalDiagLatent',
    'MixtureIndependentNormalLatent',
]


class MultivariateNormalDiagLatent(DistributionDense):
  r""" Multivariate normal diagonal latent distribution """

  def __init__(self,
               units: int,
               prior_loc: float = 0.,
               prior_scale: float = 1.,
               projection: bool = True,
               name: str = "Latents",
               **kwargs):
    super().__init__(
        event_shape=(int(units),),
        posterior=MultivariateNormalLayer,
        posterior_kwargs=dict(covariance='diag', scale_activation='softplus1'),
        prior=MultivariateNormalDiag(loc=tf.fill((units,), prior_loc),
                                     scale_identity_multiplier=prior_scale),
        projection=projection,
        name=name,
        **kwargs,
    )


class IndependentNormalLatent(DistributionDense):
  r""" Independent normal distribution latent """

  def __init__(self,
               units: int,
               prior_loc: float = 0.,
               prior_scale: float = 1.,
               projection: bool = True,
               name: str = "Latents",
               **kwargs):
    super().__init__(
        event_shape=(int(units),),
        posterior=NormalLayer,
        posterior_kwargs=dict(scale_activation='softplus1'),
        prior=Independent(Normal(loc=tf.fill((units,), prior_loc),
                                 scale=tf.fill((units,), prior_scale)),
                          reinterpreted_batch_ndims=1),
        projection=projection,
        name=name,
        **kwargs,
    )


class MixtureIndependentNormalLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, projection=True, **kwargs):
    kwargs['covariance'] = 'none'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, projection=projection, **kwargs)
    self.set_prior()


class MixtureMultivariateNormalDiagLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, projection=True, **kwargs):
    kwargs['covariance'] = 'diag'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, projection=projection, **kwargs)
    self.set_prior()
